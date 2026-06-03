from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import Location, VARIABLES
from weather_ensemble.models import ForecastRecord
from weather_ensemble.sources import FORECAST_SOURCES, open_meteo


def collect_forecasts(db_path: Path, location: Location) -> list[ForecastRecord]:
    records: list[ForecastRecord] = []
    for source_name, fetcher in FORECAST_SOURCES.items():
        try:
            records.append(fetcher(location))
        except Exception as exc:  # keep collection robust if one source fails
            print(f"WARN: {source_name} failed: {exc}")

    with db.connect(db_path) as conn:
        db.insert_forecasts(conn, records)
    return records


def record_actual(db_path: Path, location: Location, target_date: date | None = None) -> None:
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    actual = open_meteo.fetch_actual(location, target_date)
    with db.connect(db_path) as conn:
        db.upsert_actual(conn, actual)


def backfill(db_path: Path, location: Location, days_back: int) -> None:
    with db.connect(db_path) as conn:
        for i in range(1, days_back + 1):
            actual = open_meteo.fetch_actual(location, date.today() - timedelta(days=i))
            db.upsert_actual(conn, actual)

        for model in ["best_match", "bom_access_global"]:
            records = open_meteo.fetch_historical_forecasts(location, days_back, model=model)
            db.insert_forecasts(conn, records)


def load_modelling_table(db_path: Path, location: Location, window_days: int | None = None) -> pd.DataFrame:
    cutoff = "0000-01-01"
    if window_days:
        cutoff = (date.today() - timedelta(days=window_days)).isoformat()

    with db.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT
                f.source,
                f.location_name,
                f.forecast_date,
                f.collected_at,
                f.max_temp,
                f.min_temp,
                f.rain_probability,
                f.uv_index,
                f.wind_speed,
                a.max_temp AS actual_max_temp,
                a.min_temp AS actual_min_temp,
                a.rain_probability AS actual_rain_probability,
                a.uv_index AS actual_uv_index,
                a.wind_speed AS actual_wind_speed,
                a.precipitation_sum AS actual_precipitation_sum
            FROM forecasts f
            JOIN actuals a
              ON f.location_name = a.location_name
             AND f.forecast_date = a.actual_date
            WHERE f.location_name = ?
              AND f.forecast_date >= ?
            ORDER BY f.forecast_date, f.source
            """,
            conn,
            params=(location.name, cutoff),
        )


def classify_condition(rain_probability: float | None) -> str:
    if rain_probability is None:
        return "clear"
    if rain_probability < 20:
        return "clear"
    if rain_probability <= 60:
        return "cloudy"
    return "rainy"


def compute_mae_scores(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    if df.empty:
        return scores

    for source, source_df in df.groupby("source"):
        scores[source] = {}
        for var in VARIABLES:
            actual_col = f"actual_{var}"
            if var == "rain_probability":
                actual_col = "actual_rain_probability"
            if actual_col not in source_df:
                continue
            pair = source_df[[var, actual_col]].dropna()
            if not pair.empty:
                scores[source][var] = float((pair[var] - pair[actual_col]).abs().mean())
    return scores


def latest_forecasts_for_date(db_path: Path, location: Location, target_date: date) -> pd.DataFrame:
    with db.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT f.*
            FROM forecasts f
            JOIN (
                SELECT source, MAX(collected_at) AS latest_collected_at
                FROM forecasts
                WHERE location_name = ? AND forecast_date = ?
                GROUP BY source
            ) latest
              ON f.source = latest.source
             AND f.collected_at = latest.latest_collected_at
            WHERE f.location_name = ? AND f.forecast_date = ?
            ORDER BY f.source
            """,
            conn,
            params=(location.name, target_date.isoformat(), location.name, target_date.isoformat()),
        )
    return df


def blend_forecast(db_path: Path, location: Location, window_days: int) -> dict[str, Any]:
    target_date = date.today() + timedelta(days=1)
    forecast_df = latest_forecasts_for_date(db_path, location, target_date)
    if forecast_df.empty:
        return {"error": "No forecasts found. Run collect first.", "forecast_date": target_date.isoformat()}

    history_df = load_modelling_table(db_path, location, window_days=window_days)
    scores = compute_mae_scores(history_df)

    blended: dict[str, float | None] = {}
    metadata: dict[str, Any] = {"scores": scores, "sources": forecast_df["source"].tolist()}

    for var in VARIABLES:
        rows = forecast_df[["source", var]].dropna()
        if rows.empty:
            blended[var] = None
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        details = []
        for _, row in rows.iterrows():
            source = row["source"]
            prediction = float(row[var])
            mae = scores.get(source, {}).get(var)
            weight = 1.0 / mae if mae and mae > 0 else 1.0
            weighted_sum += prediction * weight
            total_weight += weight
            details.append({"source": source, "prediction": prediction, "mae": mae, "weight": weight})

        blended[var] = round(weighted_sum / total_weight, 2) if total_weight else None
        metadata[var] = details

    with db.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ensemble_predictions (
                location_name, lat, lon, forecast_date, generated_at, window_days,
                max_temp, min_temp, rain_probability, uv_index, wind_speed, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name,
                location.lat,
                location.lon,
                target_date.isoformat(),
                datetime.now().isoformat(timespec="seconds"),
                window_days,
                blended.get("max_temp"),
                blended.get("min_temp"),
                blended.get("rain_probability"),
                blended.get("uv_index"),
                blended.get("wind_speed"),
                str(metadata),
            ),
        )
        conn.commit()

    return {"forecast_date": target_date.isoformat(), "blended": blended, "metadata": metadata}


def export_modelling_table(db_path: Path, location: Location, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_modelling_table(db_path, location)
    df.to_parquet(output_path, index=False)
    return output_path
