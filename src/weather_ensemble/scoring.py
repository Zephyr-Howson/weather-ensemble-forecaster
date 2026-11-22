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
MODEL_BEST = "best"

CONTINUOUS_TARGETS = [t for t in TARGETS if t != "did_rain"]

# Live collection, blending and ML/Best prediction (collect_forecasts,
# blend_forecast, predict_latest_ml, best.py) all target the next local day
# and run once a day, so every row scored here is treated as a 1-day-ahead
# forecast and there is no lead-time bucketing.


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
                # outcome (absolute error between a 0-1 forecast probability
                # and the 0/1 actual), not a thresholded classification.
                # Callers normalize whichever probability field their model
                # carries (rain_probability, 0-100; did_rain_probability,
                # already 0-1) into the _rain_prob_fraction column.
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


def _best_predictions(db_path: Path, location: Location, window_days: int | None = None) -> list[dict]:
    """The adaptive per-target 'Best' pick (see best.py) - mirrors _ml_predictions.

    best_predictions has no did_rain column, only did_rain_probability, which
    best.py stores as a 0-1 fraction (the rain probability of the candidate it
    picked for precipitation_sum, converted from rain_probability/100 when
    that candidate is a raw source or Weighted), so no further conversion is
    needed here.
    """
    cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else "0000-01-01"
    with db.connect(db_path) as conn:
        wide = pd.read_sql_query(
            """
            WITH latest AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY location_name, forecast_date ORDER BY generated_at DESC
                ) AS rn
                FROM best_predictions
                WHERE location_name = ? AND forecast_date >= ?
            )
            SELECT b.location_name, b.forecast_date, b.max_temp, b.min_temp,
                   b.precipitation_sum, b.did_rain_probability, b.wind_speed, b.wind_gusts,
                   b.cloud_cover, b.humidity, b.pressure_msl,
                   a.max_temp AS actual_max_temp, a.min_temp AS actual_min_temp,
                   a.precipitation_sum AS actual_precipitation_sum, a.did_rain AS actual_did_rain,
                   a.wind_speed AS actual_wind_speed,
                   a.wind_gusts AS actual_wind_gusts, a.cloud_cover AS actual_cloud_cover,
                   a.humidity AS actual_humidity, a.pressure_msl AS actual_pressure_msl
            FROM latest b
            JOIN actuals a ON a.location_name = b.location_name AND a.actual_date = b.forecast_date
            WHERE b.rn = 1
            """,
            conn,
            params=(location.name, cutoff),
        )
    if wide.empty:
        return []
    if "did_rain_probability" in wide.columns:
        wide = wide.assign(_rain_prob_fraction=wide["did_rain_probability"])
    return _long_rows_from_wide(wide, model_col="", date_col="forecast_date", source_prefix=MODEL_BEST)


def _baseline_predictions(
    db_path: Path,
    location: Location,
    climatology_window: int = 30,
    climatology_min_periods: int = 7,
    window_days: int | None = None,
) -> list[dict]:
    """Persistence ("tomorrow = today") and trailing-average climatology baselines.

    Both are derived purely from the actuals history (no forecast needed), using
    only data strictly before the date being predicted, so neither can see the
    answer. Persistence is the previous actuals row's value; climatology is the
    mean of the previous `climatology_window` actuals rows (needing at least
    `climatology_min_periods` of them). Both count rows, not calendar days, so
    a missing actuals date is skipped over. Climatology is a trailing mean, not
    a day-of-year seasonal average. For did_rain, persistence is a hard 0/1 and
    climatology is the recent fraction of rainy days.

    When `window_days` is set (see build_predictions_long), actuals are fetched
    `climatology_window` extra days further back so the climatology mean for
    the earliest visible date still has its full lookback; output rows before
    the `window_days` cutoff are then dropped, so the output covers the same
    span as every other prediction source.
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

    `model` ranges over every raw forecast source, plus 'ensemble', 'ml', 'best',
    and the two baselines. This is the single table every rollup (rolling
    accuracy over time, leaderboards) in this module is built from. Adds
    `error` (predicted - actual) and `abs_error` columns.

    `window_days` bounds every underlying query to forecast dates on or after
    the location's local today minus `window_days`, so the cost of a run
    doesn't grow with total history. The accuracy report passes
    `report_history_days + report_window` (build_html_report only shows the
    last `history_days` anyway). None reads the full history (backtest.py's
    walk-forward Best backcast uses this, since its windows are relative to
    past target dates rather than today).
    """
    rows: list[dict] = []
    for location in locations:
        rows.extend(_source_predictions(db_path, location, window_days=window_days))
        rows.extend(_ensemble_predictions(db_path, location, window_days=window_days))
        rows.extend(_ml_predictions(db_path, location, window_days=window_days))
        rows.extend(_best_predictions(db_path, location, window_days=window_days))
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


def _best_predictions_period(
    db_path: Path, location: Location, period: str, window_days: int | None = None
) -> list[dict]:
    """Sub-daily rain for Best: mirrors _ml_predictions_period. best.py never
    picks a separate period winner - each row is the period forecast of the
    candidate that won that date's *daily* precipitation_sum target, so these
    scores show how that inherited choice performs at this period.
    """
    cutoff = (local_today(location) - timedelta(days=window_days)).isoformat() if window_days else "0000-01-01"
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        wide = pd.read_sql_query(
            """
            WITH latest AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY location_name, forecast_date, period ORDER BY generated_at DESC
                ) AS rn
                FROM best_predictions_periods
                WHERE location_name = ? AND period = ? AND forecast_date >= ?
            )
            SELECT b.location_name, b.forecast_date, b.precipitation_sum,
                   a.precipitation_sum AS actual_precipitation_sum
            FROM latest b
            JOIN actual_periods a
              ON a.location_name = b.location_name AND a.actual_date = b.forecast_date AND a.period = ?
            WHERE b.rn = 1
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
                "model": MODEL_BEST,
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
    """Sub-daily rain: mirrors build_predictions_long, with one synthetic target
    per period ("precipitation_sum_morning" etc.) instead of the daily TARGETS
    set. Reads the periods DB (forecast_periods, ensemble/ml/best
    _predictions_periods, actual_periods). leaderboard(), rolling_error_over_time()
    and report.py's charts are generic over `target` strings, so the CLI
    concatenates this onto the daily long_df and report.py lists these names in
    TARGET_ORDER to display them.
    """
    rows: list[dict] = []
    for location in locations:
        for period in PERIODS:
            rows.extend(_source_predictions_period(db_path, location, period, window_days=window_days))
            rows.extend(_ensemble_predictions_period(db_path, location, period, window_days=window_days))
            rows.extend(_ml_predictions_period(db_path, location, period, window_days=window_days))
            rows.extend(_best_predictions_period(db_path, location, period, window_days=window_days))
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
    date. Near the most recent date there's no "after" data yet, so
    `min_periods=1` lets the window use however many days are available rather
    than going blank - which means each date within `window // 2` days of the
    most recent one is recalculated (using more of its "after" side) every time
    this runs, until enough days have passed to fill the window on both sides.

    Each (target, model) series is reindexed onto its own full calendar date
    range (first through last appearance) before rolling, because groupby only
    yields rows for dates that have data. A date with no scored rows gets
    abs_error=NaN, which contributes nothing to any average but still receives
    a window-smoothed rolling_mae from the surrounding days. Dates before a
    model's first day or after its last are not filled. Output columns:
    forecast_date, abs_error (daily pooled mean), rolling_mae, target, model.
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
        group = group.sort_values("forecast_date").set_index("forecast_date")
        full_range = pd.date_range(group.index.min(), group.index.max(), freq="D")
        reindexed = group[["abs_error"]].reindex(full_range)
        reindexed["rolling_mae"] = reindexed["abs_error"].rolling(window=window, center=True, min_periods=1).mean()
        reindexed["target"] = target
        reindexed["model"] = model
        out.append(reindexed.rename_axis("forecast_date").reset_index())
    return pd.concat(out, ignore_index=True) if out else daily


def leaderboard(long_df: pd.DataFrame, recent_days: int | None = 30) -> pd.DataFrame:
    """MAE and row count (`mae`, `n`) per (target, model), lowest MAE first per target.

    Only rows with forecast_date in the last `recent_days` days before the
    latest date in `long_df` are counted (None counts every row). Rows from all
    locations in `long_df` are pooled.

    For 'did_rain' this is the mean absolute error between a 0-1 probability
    and the 0/1 outcome (for the persistence baseline, whose prediction is a
    hard 0/1, it is the plain error rate). Lower is better for every target.
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
