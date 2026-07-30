from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import (
    PERIODS,
    TARGETS,
    Location,
    get_periods_db_path,
    local_today,
)
from weather_ensemble.service import load_modelling_table, load_period_modelling_table

BASELINE_PERSISTENCE = "baseline_persistence"
BASELINE_CLIMATOLOGY = "baseline_climatology"
MODEL_ENSEMBLE = "ensemble"
MODEL_ML = "ml"

CONTINUOUS_TARGETS = [t for t in TARGETS if t != "did_rain"]

# Every prediction in this project (raw source, ensemble, ML) is made exactly
# one day ahead of the date it's for - collect_forecasts/blend_forecast/
# predict_latest_ml all target "tomorrow" and run once a day. So unlike a
# typical forecast-accuracy setup, there's no lead-time bucketing to worry
# about here: every row scored below is a same lead-time (1-day-ahead) forecast.


def _long_rows_from_wide(
    wide: pd.DataFrame, model_col: str, date_col: str, source_prefix: str | None = None
) -> list[dict]:
    """Melt a wide (one row per prediction) frame into long predicted/actual rows."""
    rows: list[dict] = []
    for _, r in wide.iterrows():
        model = r[model_col] if source_prefix is None else source_prefix
        for target in TARGETS:
            actual_col = f"actual_{target}"
            if actual_col not in wide.columns:
                continue
            actual = r.get(actual_col)
            if pd.isna(actual):
                continue

            if target == "did_rain":
                # Scored as % chance of rain against the binary rain/no-rain
                # outcome (mean absolute error between a 0-1 forecast
                # probability and the 0/1 actual), not a thresholded
                # classification - callers normalize whichever probability
                # field their model carries (rain_probability, 0-100;
                # did_rain_probability, already 0-1) into this common column.
                predicted = r.get("_rain_prob_fraction")
                if pd.isna(predicted):
                    continue
                predicted = float(predicted)
            else:
                predicted = r.get(target)
                if pd.isna(predicted):
                    continue
                predicted = float(predicted)

            rows.append(
                {
                    "model": model,
                    "location_name": r["location_name"],
                    "forecast_date": r[date_col],
                    "target": target,
                    "predicted": predicted,
                    "actual": float(actual),
                }
            )
    return rows


def _source_predictions(db_path: Path, location: Location, window_days: int | None = None) -> list[dict]:
    wide = load_modelling_table(db_path, location, window_days=window_days)
    if wide.empty:
        return []
    if "rain_probability" in wide.columns:
        wide = wide.assign(_rain_prob_fraction=wide["rain_probability"] / 100.0)
    return _long_rows_from_wide(wide, model_col="source", date_col="forecast_date")


def _ensemble_predictions(db_path: Path, location: Location, window_days: int | None = None) -> list[dict]:
    cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else "0000-01-01"
    with db.connect(db_path) as conn:
        wide = pd.read_sql_query(
            """
            WITH latest AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY location_name, forecast_date ORDER BY generated_at DESC
                ) AS rn
                FROM ensemble_predictions
                WHERE location_name = ? AND forecast_date >= ?
            )
            SELECT e.location_name, e.forecast_date, e.max_temp, e.min_temp,
                   e.precipitation_sum, e.did_rain, e.rain_probability, e.wind_speed, e.wind_gusts,
                   e.cloud_cover, e.humidity, e.pressure_msl,
                   a.max_temp AS actual_max_temp, a.min_temp AS actual_min_temp,
                   a.precipitation_sum AS actual_precipitation_sum, a.did_rain AS actual_did_rain,
                   a.wind_speed AS actual_wind_speed,
                   a.wind_gusts AS actual_wind_gusts, a.cloud_cover AS actual_cloud_cover,
                   a.humidity AS actual_humidity, a.pressure_msl AS actual_pressure_msl
            FROM latest e
            JOIN actuals a ON a.location_name = e.location_name AND a.actual_date = e.forecast_date
            WHERE e.rn = 1
            """,
            conn,
            params=(location.name, cutoff),
        )
    if wide.empty:
        return []
    if "rain_probability" in wide.columns:
        wide = wide.assign(_rain_prob_fraction=wide["rain_probability"] / 100.0)
    return _long_rows_from_wide(wide, model_col="", date_col="forecast_date", source_prefix=MODEL_ENSEMBLE)


def _ml_predictions(db_path: Path, location: Location, window_days: int | None = None) -> list[dict]:
    cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else "0000-01-01"
    with db.connect(db_path) as conn:
        wide = pd.read_sql_query(
            """
            WITH latest AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY location_name, forecast_date ORDER BY generated_at DESC
                ) AS rn
                FROM ml_predictions
                WHERE location_name = ? AND forecast_date >= ?
            )
            SELECT m.location_name, m.forecast_date, m.max_temp, m.min_temp,
                   m.precipitation_sum, m.did_rain, m.did_rain_probability, m.wind_speed, m.wind_gusts,
                   m.cloud_cover, m.humidity, m.pressure_msl,
                   a.max_temp AS actual_max_temp, a.min_temp AS actual_min_temp,
                   a.precipitation_sum AS actual_precipitation_sum, a.did_rain AS actual_did_rain,
                   a.wind_speed AS actual_wind_speed,
                   a.wind_gusts AS actual_wind_gusts, a.cloud_cover AS actual_cloud_cover,
                   a.humidity AS actual_humidity, a.pressure_msl AS actual_pressure_msl
            FROM latest m
            JOIN actuals a ON a.location_name = m.location_name AND a.actual_date = m.forecast_date
            WHERE m.rn = 1
            """,
            conn,
            params=(location.name, cutoff),
        )
    if wide.empty:
        return []
    if "did_rain_probability" in wide.columns:
        wide = wide.assign(_rain_prob_fraction=wide["did_rain_probability"])
    return _long_rows_from_wide(wide, model_col="", date_col="forecast_date", source_prefix=MODEL_ML)


def _baseline_predictions(
    db_path: Path,
    location: Location,
    climatology_window: int = 30,
    climatology_min_periods: int = 7,
    window_days: int | None = None,
) -> list[dict]:
    """Persistence ("tomorrow = today") and trailing-average climatology baselines.

    Both are derived purely from the actuals history (no forecast needed), using
    only data strictly before the date being predicted so this can't leak the
    answer into its own baseline. With only a few months of history so far,
    climatology uses a trailing rolling mean rather than a true day-of-year
    seasonal average; once a year-plus of actuals accumulate, this window can
    be swapped for a same-day-of-year lookup.

    `window_days` (see build_predictions_long) is extended by `climatology_window`
    extra days of actuals when *fetching* - the rolling climatology mean for
    the very first visible date still needs its own trailing lookback, so
    fetching only `window_days` back would quietly shrink/degrade its rolling
    average for dates right at the edge of the window versus the unbounded
    read it replaces. Those extra older rows exist purely to feed that rolling
    calculation, though - they're trimmed back off below so the *output* still
    only covers `window_days`, exactly like every other prediction source.
    """
    visible_cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else None
    fetch_cutoff = (
        (local_today(location) - timedelta(days=window_days + climatology_window)).isoformat()
        if window_days
        else "0000-01-01"
    )
    with db.connect(db_path) as conn:
        actuals = pd.read_sql_query(
            "SELECT location_name, actual_date, max_temp, min_temp, precipitation_sum, "
            "did_rain, wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl "
            "FROM actuals "
            "WHERE location_name = ? AND actual_date >= ? ORDER BY actual_date",
            conn,
            params=(location.name, fetch_cutoff),
        )
    if actuals.empty:
        return []

    rows: list[dict] = []
    for target in TARGETS:
        if target not in actuals.columns:
            continue
        series = actuals[target].astype(float)
        persistence = series.shift(1)
        climatology = series.rolling(window=climatology_window, min_periods=climatology_min_periods).mean().shift(1)

        for model, predicted_series in ((BASELINE_PERSISTENCE, persistence), (BASELINE_CLIMATOLOGY, climatology)):
            for idx, predicted in predicted_series.items():
                actual = series.iloc[idx]
                if pd.isna(predicted) or pd.isna(actual):
                    continue
                forecast_date = actuals["actual_date"].iloc[idx]
                if visible_cutoff is not None and forecast_date < visible_cutoff:
                    continue
                rows.append(
                    {
                        "model": model,
                        "location_name": location.name,
                        "forecast_date": forecast_date,
                        "target": target,
                        "predicted": float(predicted),
                        "actual": float(actual),
                    }
                )
    return rows


def build_predictions_long(db_path: Path, locations: list[Location], window_days: int | None = None) -> pd.DataFrame:
    """One row per (model, location, date, target) with a predicted/actual pair.

    `model` ranges over every raw forecast source, plus 'ensemble', 'ml', and the
    two baselines. This is the single table every rollup (rolling accuracy over
    time, leaderboards) in this module is built from.

    `window_days` bounds every underlying query to recent history instead of
    scanning every row ever collected - a real incident: with no bound here,
    report generation re-read and re-melted the *entire* history of every
    table on every run, so its cost grew a little more every single day even
    though `build_html_report` immediately discards everything older than its
    own `history_days` right after. Left as None (unbounded) for callers that
    genuinely need full history (e.g. the notebooks).
    """
    rows: list[dict] = []
    for location in locations:
        rows.extend(_source_predictions(db_path, location, window_days=window_days))
        rows.extend(_ensemble_predictions(db_path, location, window_days=window_days))
        rows.extend(_ml_predictions(db_path, location, window_days=window_days))
        rows.extend(_baseline_predictions(db_path, location, window_days=window_days))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["abs_error"] = (df["predicted"] - df["actual"]).abs()
    df["error"] = df["predicted"] - df["actual"]
    return df


def _source_predictions_period(
    db_path: Path, location: Location, period: str, window_days: int | None = None
) -> list[dict]:
    wide = load_period_modelling_table(db_path, location, period, window_days=window_days)
    if wide.empty:
        return []
    target = f"precipitation_sum_{period}"
    rows: list[dict] = []
    for _, r in wide.iterrows():
        actual = r.get("actual_precipitation_sum")
        predicted = r.get("precipitation_sum")
        if pd.isna(actual) or pd.isna(predicted):
            continue
        rows.append(
            {
                "model": r["source"],
                "location_name": r["location_name"],
                "forecast_date": r["forecast_date"],
                "target": target,
                "predicted": float(predicted),
                "actual": float(actual),
            }
        )
    return rows


def _ensemble_predictions_period(
    db_path: Path, location: Location, period: str, window_days: int | None = None
) -> list[dict]:
    cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else "0000-01-01"
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        wide = pd.read_sql_query(
            """
            WITH latest AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY location_name, forecast_date, period ORDER BY generated_at DESC
                ) AS rn
                FROM ensemble_predictions_periods
                WHERE location_name = ? AND period = ? AND forecast_date >= ?
            )
            SELECT e.location_name, e.forecast_date, e.precipitation_sum,
                   a.precipitation_sum AS actual_precipitation_sum
            FROM latest e
            JOIN actual_periods a
              ON a.location_name = e.location_name AND a.actual_date = e.forecast_date AND a.period = ?
            WHERE e.rn = 1
            """,
            conn,
            params=(location.name, period, cutoff, period),
        )
    if wide.empty:
        return []
    target = f"precipitation_sum_{period}"
    rows: list[dict] = []
    for _, r in wide.iterrows():
        if pd.isna(r["precipitation_sum"]) or pd.isna(r["actual_precipitation_sum"]):
            continue
        rows.append(
            {
                "model": MODEL_ENSEMBLE,
                "location_name": r["location_name"],
                "forecast_date": r["forecast_date"],
                "target": target,
                "predicted": float(r["precipitation_sum"]),
                "actual": float(r["actual_precipitation_sum"]),
            }
        )
    return rows


def _ml_predictions_period(
    db_path: Path, location: Location, period: str, window_days: int | None = None
) -> list[dict]:
    cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else "0000-01-01"
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        wide = pd.read_sql_query(
            """
            WITH latest AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY location_name, forecast_date, period ORDER BY generated_at DESC
                ) AS rn
                FROM ml_predictions_periods
                WHERE location_name = ? AND period = ? AND forecast_date >= ?
            )
            SELECT m.location_name, m.forecast_date, m.precipitation_sum,
                   a.precipitation_sum AS actual_precipitation_sum
            FROM latest m
            JOIN actual_periods a
              ON a.location_name = m.location_name AND a.actual_date = m.forecast_date AND a.period = ?
            WHERE m.rn = 1
            """,
            conn,
            params=(location.name, period, cutoff, period),
        )
    if wide.empty:
        return []
    target = f"precipitation_sum_{period}"
    rows: list[dict] = []
    for _, r in wide.iterrows():
        if pd.isna(r["precipitation_sum"]) or pd.isna(r["actual_precipitation_sum"]):
            continue
        rows.append(
            {
                "model": MODEL_ML,
                "location_name": r["location_name"],
                "forecast_date": r["forecast_date"],
                "target": target,
                "predicted": float(r["precipitation_sum"]),
                "actual": float(r["actual_precipitation_sum"]),
            }
        )
    return rows


def _baseline_predictions_period(
    db_path: Path,
    location: Location,
    period: str,
    climatology_window: int = 30,
    climatology_min_periods: int = 7,
    window_days: int | None = None,
) -> list[dict]:
    """Persistence/climatology baselines from actual_periods history - mirrors _baseline_predictions."""
    visible_cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else None
    fetch_cutoff = (
        (local_today(location) - timedelta(days=window_days + climatology_window)).isoformat()
        if window_days
        else "0000-01-01"
    )
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        actuals = pd.read_sql_query(
            "SELECT location_name, actual_date, precipitation_sum FROM actual_periods "
            "WHERE location_name = ? AND period = ? AND actual_date >= ? ORDER BY actual_date",
            conn,
            params=(location.name, period, fetch_cutoff),
        )
    if actuals.empty:
        return []

    target = f"precipitation_sum_{period}"
    series = actuals["precipitation_sum"].astype(float)
    persistence = series.shift(1)
    climatology = series.rolling(window=climatology_window, min_periods=climatology_min_periods).mean().shift(1)

    rows: list[dict] = []
    for model, predicted_series in ((BASELINE_PERSISTENCE, persistence), (BASELINE_CLIMATOLOGY, climatology)):
        for idx, predicted in predicted_series.items():
            actual = series.iloc[idx]
            if pd.isna(predicted) or pd.isna(actual):
                continue
            forecast_date = actuals["actual_date"].iloc[idx]
            if visible_cutoff is not None and forecast_date < visible_cutoff:
                continue
            rows.append(
                {
                    "model": model,
                    "location_name": location.name,
                    "forecast_date": forecast_date,
                    "target": target,
                    "predicted": float(predicted),
                    "actual": float(actual),
                }
            )
    return rows


def build_period_predictions_long(db_path: Path, locations: list[Location], window_days: int | None = None) -> pd.DataFrame:
    """Small-slice sub-daily rain: mirrors build_predictions_long, one
    synthetic target per period ("precipitation_sum_morning" etc.) instead of
    the daily TARGETS set. The existing report renderer is fully generic over
    `target` strings, so concatenating this with the daily long_df and adding
    these names to the target list (report.py) is enough to display them -
    no changes needed to leaderboard()/rolling_error_over_time() or any
    chart-building code below.
    """
    rows: list[dict] = []
    for location in locations:
        for period in PERIODS:
            rows.extend(_source_predictions_period(db_path, location, period, window_days=window_days))
            rows.extend(_ensemble_predictions_period(db_path, location, period, window_days=window_days))
            rows.extend(_ml_predictions_period(db_path, location, period, window_days=window_days))
            rows.extend(_baseline_predictions_period(db_path, location, period, window_days=window_days))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    df["abs_error"] = (df["predicted"] - df["actual"]).abs()
    df["error"] = df["predicted"] - df["actual"]
    return df


def rolling_error_over_time(long_df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Rolling mean absolute error per (target, model) over time, pooled across locations.

    Errors from every location on a given date are averaged first (one point per
    date), then smoothed with a centered `window`-day rolling mean - this is what
    the "accuracy over time" chart plots. Centered means a 7-day window is 3 days
    before + the day itself + 3 days after, not the trailing 7 days ending on that
    date. Near the most recent date there's no "after" data yet (it hasn't
    happened), so `min_periods=1` lets the window use however many days are
    actually available rather than going blank - which means each date within
    `window // 2` days of the most recent one is recalculated (using more of its
    "after" side) every time this runs, until enough days have passed to fill
    the window on both sides.
    """
    if long_df.empty:
        return long_df

    daily = (
        long_df.groupby(["target", "model", "forecast_date"])["abs_error"]
        .mean()
        .reset_index()
        .sort_values("forecast_date")
    )

    out = []
    for (target, model), group in daily.groupby(["target", "model"]):
        group = group.sort_values("forecast_date").copy()
        group["rolling_mae"] = group["abs_error"].rolling(window=window, center=True, min_periods=1).mean()
        out.append(group)
    return pd.concat(out, ignore_index=True) if out else daily


def leaderboard(long_df: pd.DataFrame, recent_days: int | None = 30) -> pd.DataFrame:
    """Mean absolute error per (target, model), most recent first, best (lowest) ranked first.

    For 'did_rain' this is the mean absolute error against the 0/1 outcome (a
    Brier-like score for models that emit a probability, and a plain error rate
    for models that emit a hard 0/1 class) - lower is better for every target,
    including did_rain, so all rows can be read the same way.
    """
    if long_df.empty:
        return long_df

    scoped = long_df
    if recent_days is not None:
        cutoff = long_df["forecast_date"].max() - pd.Timedelta(days=recent_days)
        scoped = long_df[long_df["forecast_date"] > cutoff]

    summary = (
        scoped.groupby(["target", "model"])["abs_error"]
        .agg(mae="mean", n="count")
        .reset_index()
        .sort_values(["target", "mae"])
    )
    return summary
