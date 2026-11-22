from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import (
    FORECAST_VARIABLES,
    OPEN_METEO_BACKFILL_MODELS,
    OPEN_METEO_MODELS,
    PERIODS,
    RAIN_THRESHOLD_MM,
    TARGETS,
    Location,
    get_periods_db_path,
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


def collect_forecasts(db_path: Path, location: Location, target_date: date | None = None) -> list[ForecastRecord]:
    """Collect every configured source's forecast for one target date (default:
    default_forecast_target_date) and insert them into `forecasts`.

    Every fetcher is given the same already-resolved date rather than working
    out "tomorrow" itself, so a run that starts after local midnight still
    collects the date the rest of the run works on. A source that fails is
    logged and skipped; the others are still stored.
    """
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)
    records: list[ForecastRecord] = []
    for source_name, fetcher in FORECAST_SOURCES.items():
        try:
            records.append(fetcher(location, target_date))
        except Exception as exc:  # noqa: BLE001 - keep collection robust if one source fails
            print(f"WARN: {source_name} failed: {_safe_error(exc)}")

    with db.connect(db_path) as conn:
        db.insert_forecasts(conn, records)
    return records


def collect_open_meteo_only(db_path: Path, location: Location, target_date: date | None = None) -> list[ForecastRecord]:
    """Collect only Open-Meteo model outputs, for the same resolved target date.

    This is useful for a completely free/no-key workflow and for debugging the
    core model ensemble without optional external APIs.
    """
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)
    records: list[ForecastRecord] = []
    for source_name, fetcher in OPEN_METEO_FORECAST_SOURCES.items():
        try:
            records.append(fetcher(location, target_date))
        except Exception as exc:  # noqa: BLE001 - keep collection robust if one source fails
            print(f"WARN: {source_name} failed: {_safe_error(exc)}")

    with db.connect(db_path) as conn:
        db.insert_forecasts(conn, records)
    return records


# How far back missing_ensemble_dates (used by cli._catch_up_missed_forecasts)
# and actual_dates_to_record look for a gap. Bounded rather than scanning all
# history - a location that's been offline/misconfigured for longer than this
# needs a deliberate --backfill, not an ever-growing walk-forward
# reconstruction tacked onto every daily run.
CATCH_UP_LOOKBACK_DAYS = 10


def default_forecast_target_date(db_path: Path, location: Location) -> date:
    """The date the collect/forecast/predict-ml/predict-best/narrate steps
    treat as "tomorrow": min(local tomorrow, the day after this location's
    latest ensemble_predictions forecast_date), or local tomorrow if the
    location has no ensemble prediction yet.

    A run that starts after local midnight (GitHub's scheduled runs can start
    hours late) would otherwise skip a day: wall-clock tomorrow would already be
    two days past the last date the pipeline predicted. With the cap, such a
    run works on the day after the last prediction instead. If that date can't
    be collected, downstream steps report that no forecast exists for it, and
    cli._catch_up_missed_forecasts backfills it on a later run.

    Not stable across a run: once blend_forecast writes the ensemble row for
    the date this returns, the answer moves forward a day whenever local
    midnight has already passed. A multi-step run resolves it once up front
    and passes that date to every step (see cli._run_for_location).
    """
    live_tomorrow = local_today(location) + timedelta(days=1)
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT MAX(forecast_date) FROM ensemble_predictions WHERE location_name = ?",
            (location.name,),
        ).fetchone()
    if not row or not row[0]:
        return live_tomorrow
    expected = date.fromisoformat(row[0]) + timedelta(days=1)
    return min(expected, live_tomorrow)


def missing_ensemble_dates(db_path: Path, location: Location) -> list[date]:
    """Every date in the trailing CATCH_UP_LOOKBACK_DAYS (up to, not
    including, today) with no ensemble_predictions row for this location.

    Scans the whole recent window rather than just checking MAX(forecast_date):
    a later date can already exist in the table while an earlier one is
    missing, and a MAX-based check would not see that gap.

    Returns [] for a location with no ensemble_predictions history at all -
    a brand-new location's first-ever run has no "gap" to speak of, and
    should go through --backfill/--deploy-phases deliberately rather than
    have every date in the lookback window treated as missing.
    """
    today = local_today(location)
    window_start = today - timedelta(days=CATCH_UP_LOOKBACK_DAYS)
    with db.connect(db_path) as conn:
        has_any_history = (
            conn.execute(
                "SELECT 1 FROM ensemble_predictions WHERE location_name = ? LIMIT 1", (location.name,)
            ).fetchone()
            is not None
        )
        if not has_any_history:
            return []
        rows = conn.execute(
            "SELECT DISTINCT forecast_date FROM ensemble_predictions "
            "WHERE location_name = ? AND forecast_date >= ? AND forecast_date < ?",
            (location.name, window_start.isoformat(), today.isoformat()),
        ).fetchall()
    existing = {date.fromisoformat(r[0]) for r in rows}
    return [d for d in (window_start + timedelta(days=i) for i in range(CATCH_UP_LOOKBACK_DAYS)) if d not in existing]


def actual_dates_to_record(conn, table: str, location: Location) -> list[date]:
    """Yesterday, then any other day in the trailing CATCH_UP_LOOKBACK_DAYS
    with no row in `table` (actuals or actual_periods) for this location yet.

    Recording only wall-clock yesterday would skip a day whenever consecutive
    runs land on opposite sides of local midnight (23:30 on the 25th records
    the 24th, 00:30 on the 27th records the 26th, and the 25th is never
    recorded), which happens because GitHub's scheduled runs can start hours
    late. Yesterday always comes first and is always re-fetched; a location
    with no rows at all gets just yesterday, like missing_ensemble_dates,
    rather than a lookback-sized backfill.
    """
    today = local_today(location)
    yesterday = today - timedelta(days=1)
    has_any_history = (
        conn.execute(f"SELECT 1 FROM {table} WHERE location_name = ? LIMIT 1", (location.name,)).fetchone() is not None
    )
    if not has_any_history:
        return [yesterday]
    window_start = today - timedelta(days=CATCH_UP_LOOKBACK_DAYS)
    rows = conn.execute(
        f"SELECT DISTINCT actual_date FROM {table} WHERE location_name = ? AND actual_date >= ? AND actual_date < ?",
        (location.name, window_start.isoformat(), today.isoformat()),
    ).fetchall()
    existing = {date.fromisoformat(r[0]) for r in rows}
    gaps = [d for d in (window_start + timedelta(days=i) for i in range(CATCH_UP_LOOKBACK_DAYS)) if d not in existing]
    return [yesterday, *(d for d in gaps if d != yesterday)]


def record_actual(db_path: Path, location: Location, target_date: date | None = None) -> list[date]:
    """Record target_date's observed weather (default: yesterday's, plus any
    recent day still missing one - see actual_dates_to_record). Returns the
    dates recorded."""
    if target_date is not None:
        dates = [target_date]
    else:
        with db.connect(db_path) as conn:
            dates = actual_dates_to_record(conn, "actuals", location)
    for d in dates:
        actual = open_meteo.fetch_actual(location, d)
        with db.connect(db_path) as conn:
            db.upsert_actual(conn, actual)
    return dates


def backfill(db_path: Path, location: Location, days_back: int) -> None:
    """Backfill the last `days_back` days (local yesterday back to today -
    days_back) of actuals and Open-Meteo historical forecasts, daily and
    sub-daily rain periods together.

    Actuals are fetched day by day, then actual periods for the whole range,
    then each OPEN_METEO_BACKFILL_MODELS model's daily forecasts and periods
    from a single API call (fetch_historical_forecast_with_periods), so a
    source's daily precipitation total matches the sum of its own periods.
    Each fetch is isolated: a failure is logged and the rest continue.
    Open-Meteo is the only backfilled source because its Historical Forecast
    API returns archived forecasts; the optional providers are live-only.
    """
    today = local_today(location)
    with db.connect(db_path) as conn, db.connect_periods(get_periods_db_path(db_path)) as period_conn:
        for i in range(1, days_back + 1):
            try:
                actual = open_meteo.fetch_actual(location, today - timedelta(days=i))
                db.upsert_actual(conn, actual)
            except Exception as exc:  # noqa: BLE001 - one bad day shouldn't abort the whole backfill
                print(f"WARN: actual backfill failed for day -{i}: {_safe_error(exc)}")

        try:
            actual_periods = open_meteo.fetch_historical_actual_periods(location, days_back)
            for r in actual_periods:
                db.upsert_actual_period(period_conn, r)
        except Exception as exc:  # noqa: BLE001 - actuals failing shouldn't abort the model backfill below
            print(f"WARN: actual period backfill failed: {_safe_error(exc)}")

        for model in OPEN_METEO_BACKFILL_MODELS:
            try:
                forecasts, periods = open_meteo.fetch_historical_forecast_with_periods(location, days_back, model=model)
                inserted = db.insert_forecasts(conn, forecasts)
                inserted_periods = db.insert_forecast_periods(period_conn, periods)
                print(f"Backfilled {inserted} rows ({inserted_periods} period rows) for open_meteo_{model}")
            except Exception as exc:  # noqa: BLE001 - one bad model shouldn't abort the whole backfill
                print(f"WARN: historical backfill failed for open_meteo_{model}: {_safe_error(exc)}")


def collect_forecast_periods(db_path: Path, location: Location, target_date: date | None = None) -> int:
    """Live sub-daily rain collection from every OPEN_METEO_MODELS model, into
    forecast_periods. Returns the number of rows inserted.

    Open-Meteo only (not the 7 optional providers). Callers pass the same
    resolved target date as collect_forecasts, so the periods belong to the
    same date as the daily rows.
    """
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)
    records = []
    for model in OPEN_METEO_MODELS:
        try:
            records.extend(open_meteo.fetch_forecast_with_periods(location, target_date, model=model)[1])
        except Exception as exc:  # noqa: BLE001 - keep collection robust if one model fails
            print(f"WARN: open_meteo_{model} periods failed: {_safe_error(exc)}")

    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        return db.insert_forecast_periods(conn, records)


def record_actual_periods(db_path: Path, location: Location, target_date: date | None = None) -> list[date]:
    """Mirrors record_actual, for sub-daily rain actuals."""
    periods_db_path = get_periods_db_path(db_path)
    if target_date is not None:
        dates = [target_date]
    else:
        with db.connect_periods(periods_db_path) as conn:
            dates = actual_dates_to_record(conn, "actual_periods", location)
    for d in dates:
        records = open_meteo.fetch_actual_periods(location, d)
        with db.connect_periods(periods_db_path) as conn:
            for r in records:
                db.upsert_actual_period(conn, r)
    return dates


def load_modelling_table(db_path: Path, location: Location, window_days: int | None = None) -> pd.DataFrame:
    """One row per (source, forecast_date) that has an actual, with every forecast
    variable alongside its actual_<target> columns.

    Where a source has several forecasts for a date, a live-collected row wins
    over a backfilled one, then the most recently collected. `window_days`
    keeps forecast dates within that many days of local today (None: all).
    """
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
        "wind_speed": "actual_wind_speed",
        "wind_gusts": "actual_wind_gusts",
        "cloud_cover": "actual_cloud_cover",
        "humidity": "actual_humidity",
        "pressure_msl": "actual_pressure_msl",
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
    """Each source's most recently collected forecast row for target_date."""
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

    # Continuous variables with an observed counterpart: weight = 1/MAE, or 1.0
    # for a source with no (or zero) MAE in the scoring window.
    blendable = [
        "max_temp", "min_temp", "precipitation_sum", "wind_speed", "wind_gusts",
        "cloud_cover", "humidity", "pressure_msl",
    ]
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


def blend_forecast(db_path: Path, location: Location, window_days: int, target_date: date | None = None) -> dict[str, Any]:
    """Blend every source's latest forecast for target_date (default:
    default_forecast_target_date) and insert it into ensemble_predictions.

    MAE weights come from the last `window_days` days of history (counted back
    from local today), restricted to forecast dates strictly before
    target_date, so a prediction for a past date can't see data it wouldn't
    have had at the time - the same rule as the walk-forward backtest.
    Returns an "error" result, writing nothing, if no forecasts exist for
    target_date.
    """
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)
    forecast_df = latest_forecasts_for_date(db_path, location, target_date)
    if forecast_df.empty:
        return {"error": "No forecasts found. Run collect first.", "forecast_date": target_date.isoformat()}

    history_df = load_modelling_table(db_path, location, window_days=window_days)
    history_df = history_df[pd.to_datetime(history_df["forecast_date"]) < pd.Timestamp(target_date)]
    scores = compute_mae_scores(history_df)
    blended, metadata = blend_weighted(forecast_df, scores)

    with db.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ensemble_predictions (
                location_name, lat, lon, forecast_date, generated_at, window_days,
                max_temp, min_temp, rain_probability, precipitation_sum, did_rain,
                wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name, location.lat, location.lon, target_date.isoformat(),
                datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"), window_days,
                blended.get("max_temp"), blended.get("min_temp"), blended.get("rain_probability"),
                blended.get("precipitation_sum"), blended.get("did_rain"),
                blended.get("wind_speed"), blended.get("wind_gusts"), blended.get("cloud_cover"),
                blended.get("humidity"), blended.get("pressure_msl"),
                None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
            ),
        )
        conn.commit()

    return {"forecast_date": target_date.isoformat(), "blended": blended, "metadata": metadata}


def load_period_modelling_table(
    db_path: Path, location: Location, period: str, window_days: int | None = None
) -> pd.DataFrame:
    """forecast_periods joined to actual_periods for one period - mirrors
    load_modelling_table (same live-over-backfill ranking and window), for the
    one variable periods track (precipitation_sum).
    """
    cutoff = "0000-01-01"
    if window_days:
        cutoff = (local_today(location) - timedelta(days=window_days)).isoformat()

    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        return pd.read_sql_query(
            """
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY location_name, forecast_date, period, source
                           ORDER BY CASE WHEN collection_method = 'live' THEN 0 ELSE 1 END,
                                    collected_at DESC
                       ) AS rn
                FROM forecast_periods
                WHERE location_name = ? AND period = ? AND forecast_date >= ?
            )
            SELECT
                f.source, f.location_name, f.forecast_date, f.period, f.collected_at,
                f.precipitation_sum, f.rain_probability,
                a.precipitation_sum AS actual_precipitation_sum, a.did_rain AS actual_did_rain
            FROM ranked f
            JOIN actual_periods a
              ON f.location_name = a.location_name
             AND f.forecast_date = a.actual_date
             AND f.period = a.period
            WHERE f.rn = 1
            ORDER BY f.forecast_date, f.source
            """,
            conn,
            params=(location.name, period, cutoff),
        )


def compute_period_mae_scores(df: pd.DataFrame) -> dict[str, float]:
    scores: dict[str, float] = {}
    if df.empty:
        return scores
    for source, source_df in df.groupby("source"):
        pair = source_df[["precipitation_sum", "actual_precipitation_sum"]].dropna()
        if not pair.empty:
            scores[source] = float((pair["precipitation_sum"] - pair["actual_precipitation_sum"]).abs().mean())
    return scores


def latest_forecast_periods_for_date(db_path: Path, location: Location, period: str, target_date: date) -> pd.DataFrame:
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        return pd.read_sql_query(
            """
            SELECT f.*
            FROM forecast_periods f
            JOIN (
                SELECT source, MAX(collected_at) AS latest_collected_at
                FROM forecast_periods
                WHERE location_name = ? AND forecast_date = ? AND period = ?
                GROUP BY source
            ) latest
              ON f.source = latest.source
             AND f.collected_at = latest.latest_collected_at
            WHERE f.location_name = ? AND f.forecast_date = ? AND period = ?
            ORDER BY f.source
            """,
            conn,
            params=(location.name, target_date.isoformat(), period, location.name, target_date.isoformat(), period),
        )


def blend_period_precipitation(forecast_df: pd.DataFrame, scores: dict[str, float]) -> tuple[float | None, dict[str, Any]]:
    """Inverse-MAE weighted average of precipitation_sum for one (location,
    date, period) - same weighting formula as blend_weighted, simplified to a
    single variable since periods only ever predict precipitation.
    """
    metadata: dict[str, Any] = {"scores": scores, "sources": forecast_df["source"].tolist()}
    rows = forecast_df[["source", "precipitation_sum"]].dropna()
    if rows.empty:
        return None, metadata

    weighted_sum = 0.0
    total_weight = 0.0
    details = []
    for _, row in rows.iterrows():
        source = row["source"]
        prediction = float(row["precipitation_sum"])
        mae = scores.get(source)
        weight = 1.0 / mae if mae and mae > 0 else 1.0
        weighted_sum += prediction * weight
        total_weight += weight
        details.append({"source": source, "prediction": prediction, "mae": mae, "weight": weight})
    metadata["details"] = details

    blended = round(weighted_sum / total_weight, 2) if total_weight else None
    return blended, metadata


def blend_forecast_period(
    db_path: Path, location: Location, period: str, window_days: int, target_date: date | None = None
) -> dict[str, Any]:
    """Blend every source's forecast for one period of target_date (default:
    default_forecast_target_date) into ensemble_predictions_periods - mirrors
    blend_forecast."""
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)
    forecast_df = latest_forecast_periods_for_date(db_path, location, period, target_date)
    if forecast_df.empty:
        return {
            "error": "No forecast periods found. Run collect_forecast_periods first.",
            "forecast_date": target_date.isoformat(),
            "period": period,
        }

    history_df = load_period_modelling_table(db_path, location, period, window_days=window_days)
    history_df = history_df[pd.to_datetime(history_df["forecast_date"]) < pd.Timestamp(target_date)]
    scores = compute_period_mae_scores(history_df)
    blended, metadata = blend_period_precipitation(forecast_df, scores)

    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ensemble_predictions_periods (
                location_name, lat, lon, forecast_date, period, generated_at, window_days,
                precipitation_sum, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name, location.lat, location.lon, target_date.isoformat(), period,
                datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"), window_days,
                blended,
                None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
            ),
        )
        conn.commit()

    return {"forecast_date": target_date.isoformat(), "period": period, "precipitation_sum": blended, "metadata": metadata}


_RECONCILE_TABLES = {
    "ensemble": ("ensemble_predictions", "ensemble_predictions_periods"),
    "ml": ("ml_predictions", "ml_predictions_periods"),
}


def reconcile_period_predictions(
    db_path: Path, location: Location, model_kind: str, forecast_dates: list[str]
) -> dict[str, int]:
    """Rescale each date's 4 period predictions so they sum to the daily
    prediction for the same (model, location, date).

    The daily precipitation_sum and its 4-period split are blended (ensemble)
    or trained (ML) independently, with nothing forcing them to agree, and
    they routinely don't. This is a post-hoc rescale, not a re-blend: the
    latest daily prediction is the anchor and the latest row for each period
    is updated in place, scaled to match it while keeping their relative
    shape. If every period is 0 but the daily total isn't, the total is split
    evenly. Dates without a daily value or without all 4 periods are skipped.

    `model_kind` is "ensemble" or "ml". Safe to call repeatedly -
    already-reconciled dates rescale by a factor of ~1.0.
    """
    daily_table, period_table = _RECONCILE_TABLES[model_kind]
    reconciled = 0
    skipped_no_daily = 0
    skipped_incomplete_periods = 0

    with db.connect(db_path) as conn, db.connect_periods(get_periods_db_path(db_path)) as period_conn:
        for forecast_date in forecast_dates:
            daily_row = conn.execute(
                f"SELECT precipitation_sum FROM {daily_table} WHERE location_name=? AND forecast_date=? "
                "ORDER BY generated_at DESC LIMIT 1",
                (location.name, forecast_date),
            ).fetchone()
            if daily_row is None or daily_row[0] is None:
                skipped_no_daily += 1
                continue
            daily_value = daily_row[0]

            period_rows: dict[str, tuple[int, float]] = {}
            for period in PERIODS:
                row = period_conn.execute(
                    f"SELECT id, precipitation_sum FROM {period_table} "
                    "WHERE location_name=? AND forecast_date=? AND period=? "
                    "ORDER BY generated_at DESC LIMIT 1",
                    (location.name, forecast_date, period),
                ).fetchone()
                if row is None or row[1] is None:
                    break
                period_rows[period] = (row[0], row[1])
            if len(period_rows) != len(PERIODS):
                skipped_incomplete_periods += 1
                continue

            total = sum(value for _, value in period_rows.values())
            if total > 0:
                scale = daily_value / total
                new_values = {p: round(value * scale, 2) for p, (_, value) in period_rows.items()}
            elif daily_value > 0:
                # No relative shape to preserve (every period predicted 0) -
                # spread the daily total evenly rather than leave it unreconciled.
                new_values = dict.fromkeys(period_rows, round(daily_value / len(PERIODS), 2))
            else:
                new_values = {p: value for p, (_, value) in period_rows.items()}  # both ~0, already consistent

            for period, (row_id, _) in period_rows.items():
                period_conn.execute(
                    f"UPDATE {period_table} SET precipitation_sum=? WHERE id=?",
                    (new_values[period], row_id),
                )
            reconciled += 1
        period_conn.commit()

    return {
        "reconciled": reconciled,
        "skipped_no_daily": skipped_no_daily,
        "skipped_incomplete_periods": skipped_incomplete_periods,
    }


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
