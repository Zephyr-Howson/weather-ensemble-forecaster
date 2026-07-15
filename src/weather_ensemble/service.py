from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import (
    FORECAST_VARIABLES,
    Location,
    OPEN_METEO_BACKFILL_MODELS,
    RAIN_THRESHOLD_MM,
    TARGETS,
    local_today,
)
from weather_ensemble.models import ForecastRecord
from weather_ensemble.sources import FORECAST_SOURCES, OPEN_METEO_FORECAST_SOURCES, open_meteo

# requests' HTTPError messages embed the full request URL, and most providers here
# put their API key in the query string - so printing an exception verbatim can leak
# a live key into logs/CI output. Strip query strings before anything gets printed.
_QUERY_STRING = re.compile(r"\?\S*")


def _safe_error(exc: Exception) -> str:
    return _QUERY_STRING.sub("?<redacted>", str(exc))


def collect_forecasts(db_path: Path, location: Location) -> list[ForecastRecord]:
    records: list[ForecastRecord] = []
    for source_name, fetcher in FORECAST_SOURCES.items():
        try:
            records.append(fetcher(location))
        except Exception as exc:  # keep collection robust if one source fails
            print(f"WARN: {source_name} failed: {_safe_error(exc)}")

    with db.connect(db_path) as conn:
        db.insert_forecasts(conn, records)
    return records


def collect_open_meteo_only(db_path: Path, location: Location) -> list[ForecastRecord]:
    """Collect only Open-Meteo model outputs for tomorrow.

    This is useful for a completely free/no-key workflow and for debugging the
    core model ensemble without optional external APIs.
    """
    records: list[ForecastRecord] = []
    for source_name, fetcher in OPEN_METEO_FORECAST_SOURCES.items():
        try:
            records.append(fetcher(location))
        except Exception as exc:
            print(f"WARN: {source_name} failed: {_safe_error(exc)}")

    with db.connect(db_path) as conn:
        db.insert_forecasts(conn, records)
    return records


def record_actual(db_path: Path, location: Location, target_date: date | None = None) -> None:
    if target_date is None:
        target_date = local_today(location) - timedelta(days=1)
    actual = open_meteo.fetch_actual(location, target_date)
    with db.connect(db_path) as conn:
        db.upsert_actual(conn, actual)


def backfill(db_path: Path, location: Location, days_back: int) -> None:
    """Backfill actuals plus all configured Open-Meteo historical forecasts.

    Open-Meteo is intentionally the backbone here because its Historical
    Forecast API lets us retrieve archived forecasts rather than just observed
    weather. Optional providers are live-only and are not backfilled here.
    """
    today = local_today(location)
    with db.connect(db_path) as conn:
        for i in range(1, days_back + 1):
            try:
                actual = open_meteo.fetch_actual(location, today - timedelta(days=i))
                db.upsert_actual(conn, actual)
            except Exception as exc:
                print(f"WARN: actual backfill failed for day -{i}: {_safe_error(exc)}")

        for model in OPEN_METEO_BACKFILL_MODELS:
            try:
                records = open_meteo.fetch_historical_forecasts(location, days_back, model=model)
                inserted = db.insert_forecasts(conn, records)
                print(f"Backfilled {inserted} rows for open_meteo_{model}")
            except Exception as exc:
                print(f"WARN: historical backfill failed for open_meteo_{model}: {_safe_error(exc)}")


def load_modelling_table(db_path: Path, location: Location, window_days: int | None = None) -> pd.DataFrame:
    cutoff = "0000-01-01"
    if window_days:
        cutoff = (local_today(location) - timedelta(days=window_days)).isoformat()

    forecast_cols = ",\n                ".join(f"f.{var}" for var in FORECAST_VARIABLES)
    actual_cols = ",\n                ".join(f"a.{target} AS actual_{target}" for target in TARGETS)

    with db.connect(db_path) as conn:
        return pd.read_sql_query(
            f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY location_name, forecast_date, source
                           ORDER BY CASE WHEN collection_method = 'live' THEN 0 ELSE 1 END,
                                    collected_at DESC
                       ) AS rn
                FROM forecasts
                WHERE location_name = ?
                  AND forecast_date >= ?
            )
            SELECT
                f.source,
                f.location_name,
                f.forecast_date,
                f.collected_at,
                {forecast_cols},
                {actual_cols}
            FROM ranked f
            JOIN actuals a
              ON f.location_name = a.location_name
             AND f.forecast_date = a.actual_date
            WHERE f.rn = 1
            ORDER BY f.forecast_date, f.source
            """,
            conn,
            params=(location.name, cutoff),
        )


def compute_mae_scores(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    if df.empty:
        return scores

    # Score forecast variables only where a true observed target exists. Rain
    # probability is skipped because the observed outcome is did_rain/amount.
    comparable = {
        "max_temp": "actual_max_temp",
        "min_temp": "actual_min_temp",
        "precipitation_sum": "actual_precipitation_sum",
        "uv_index": "actual_uv_index",
        "wind_speed": "actual_wind_speed",
        "wind_gusts": "actual_wind_gusts",
    }

    for source, source_df in df.groupby("source"):
        scores[source] = {}
        for var, actual_col in comparable.items():
            if var not in source_df or actual_col not in source_df:
                continue
            pair = source_df[[var, actual_col]].dropna()
            if not pair.empty:
                scores[source][var] = float((pair[var] - pair[actual_col]).abs().mean())
    return scores


def latest_forecasts_for_date(db_path: Path, location: Location, target_date: date) -> pd.DataFrame:
    with db.connect(db_path) as conn:
        return pd.read_sql_query(
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


def blend_weighted(forecast_df: pd.DataFrame, scores: dict[str, dict[str, float]]) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Inverse-MAE weighted average of every source's forecast, for one target date.

    Shared by the live "tomorrow" blend (`blend_forecast`) and the historical
    walk-forward backtest (`backtest.py`) - both need the exact same weighting
    formula, just fed a forecast row and a scores dict computed differently
    (relative to real "today" vs. relative to an arbitrary past date).
    """
    blended: dict[str, float | None] = {}
    metadata: dict[str, Any] = {"scores": scores, "sources": forecast_df["source"].tolist()}

    # Weighted baseline is still useful for directly comparable continuous vars.
    blendable = ["max_temp", "min_temp", "precipitation_sum", "uv_index", "wind_speed", "wind_gusts"]
    for var in blendable:
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

    # Rain probability remains an input/consumer-facing forecast; average it rather than scoring it against a fake actual.
    if "rain_probability" in forecast_df:
        blended["rain_probability"] = round(float(forecast_df["rain_probability"].dropna().mean()), 2) if forecast_df["rain_probability"].notna().any() else None

    # Derived observed-style target from blended precipitation.
    precip = blended.get("precipitation_sum")
    blended["did_rain"] = int(precip >= RAIN_THRESHOLD_MM) if precip is not None else None

    return blended, metadata


def blend_forecast(db_path: Path, location: Location, window_days: int) -> dict[str, Any]:
    target_date = local_today(location) + timedelta(days=1)
    forecast_df = latest_forecasts_for_date(db_path, location, target_date)
    if forecast_df.empty:
        return {"error": "No forecasts found. Run collect first.", "forecast_date": target_date.isoformat()}

    history_df = load_modelling_table(db_path, location, window_days=window_days)
    scores = compute_mae_scores(history_df)
    blended, metadata = blend_weighted(forecast_df, scores)

    with db.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ensemble_predictions (
                location_name, lat, lon, forecast_date, generated_at, window_days,
                max_temp, min_temp, rain_probability, precipitation_sum, did_rain,
                uv_index, wind_speed, wind_gusts, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name, location.lat, location.lon, target_date.isoformat(),
                datetime.now().isoformat(timespec="seconds"), window_days,
                blended.get("max_temp"), blended.get("min_temp"), blended.get("rain_probability"),
                blended.get("precipitation_sum"), blended.get("did_rain"), blended.get("uv_index"),
                blended.get("wind_speed"), blended.get("wind_gusts"), json.dumps(metadata),
            ),
        )
        conn.commit()

    return {"forecast_date": target_date.isoformat(), "blended": blended, "metadata": metadata}


def export_modelling_table(db_path: Path, location: Location, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = load_modelling_table(db_path, location)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)
    return output_path


def classify_condition(rain_probability: float | None) -> str:
    """Simple consumer-facing condition bucket from forecast rain probability."""
    if rain_probability is None:
        return "unknown"
    if rain_probability < 20:
        return "clear"
    if rain_probability <= 60:
        return "cloudy"
    return "rainy"
