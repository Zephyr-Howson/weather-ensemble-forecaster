from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import (
    OPEN_METEO_MODELS,
    PERIODS,
    TARGETS,
    Location,
    get_periods_db_path,
    local_today,
)
from weather_ensemble.scoring import (
    BASELINE_CLIMATOLOGY,
    BASELINE_PERSISTENCE,
    MODEL_BEST,
    MODEL_ENSEMBLE,
    MODEL_ML,
    build_predictions_long,
)
from weather_ensemble.service import default_forecast_target_date, latest_forecasts_for_date

# "Best" picks, independently per target, whichever candidate - a raw
# source, the Weighted blend, or the ML model - had the lowest MAE for that
# one target in this location recently. The two naive baselines
# (persistence/climatology) are deliberately not candidates: they exist to
# show how much better real forecasting is than doing nothing, not as
# strategies "Best" should ever adopt - if persistence really is winning,
# that's a fact worth seeing on the baseline's own dashed line, not a value
# worth copying into a hero series. MODEL_BEST itself is also excluded - once
# report.py wired Best's own history into build_predictions_long (so it can
# appear on the leaderboard/trend charts), it became just another row in the
# same long_df this candidate pool is built from, and without this exclusion
# Best could nominate itself as a candidate for its own future selection.
# That's not just conceptually circular - _candidate_value_live/_backtest_best
# have no defined value to copy for "model == best" (it isn't a real source,
# ensemble, or ml), so on the live path it would silently win and then
# resolve to no value at all, leaving that target unfilled.
_EXCLUDED_CANDIDATES = {BASELINE_PERSISTENCE, BASELINE_CLIMATOLOGY, MODEL_BEST}

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MIN_DAYS = 14

# The daily TARGETS set. Every target except precipitation_sum/did_rain is
# selected independently (see _predictions_for_date). Sub-daily rain periods
# (precipitation_sum_{period}) are NOT an independent selection here:
# whichever candidate wins the daily precipitation_sum target has its own
# period breakdown copied too (see _period_values_for_model), so the four
# periods always sum to Best's own daily total instead of each period
# potentially picking a different winner and disagreeing with it.
BEST_TARGETS: list[str] = list(TARGETS)

# Only candidates that actually report sub-daily rain periods: the 4 raw
# Open-Meteo model variants (see collect_forecast_periods in service.py - no
# other raw provider ever collects period-level precipitation at all), plus
# the two blended models, which are themselves built from a mix of raw
# sources that always includes those 4. precipitation_sum and did_rain are
# both selected from this restricted pool (see _select_precip_and_rain), so
# whichever one wins is guaranteed to have real period data to copy - Best
# never goes blank for a target date just because the winner happened to be a
# source that never reports periods at all (BOM, WeatherAPI, OpenWeatherMap,
# Weatherbit, AccuWeather, Visual Crossing, wttr.in, SILO).
PERIOD_AWARE_CANDIDATES = {MODEL_ENSEMBLE, MODEL_ML} | {f"open_meteo_{m}" for m in OPEN_METEO_MODELS}


def candidate_pool(long_df: pd.DataFrame) -> pd.DataFrame:
    """Every real forecast strategy for one location - every raw source, plus
    Weighted and ML - with the two naive baselines filtered out.
    """
    if long_df.empty:
        return long_df
    return long_df[~long_df["model"].isin(_EXCLUDED_CANDIDATES)]


def select_best_model(
    pool: pd.DataFrame,
    target: str,
    target_date: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_days: int = DEFAULT_MIN_DAYS,
) -> tuple[str | None, dict[str, Any]]:
    """Whichever candidate had the lowest MAE for `target` over the
    `window_days` days strictly before `target_date`, requiring at least
    `min_days` scored days in that window to be eligible - a candidate with
    only a handful of lucky days shouldn't win on too small a sample.

    Only ever looks at data before target_date, so this is safe to call for
    both a live "tomorrow" prediction and a historical walk-forward backcast:
    neither can see anything it "shouldn't know yet" at that point in time.
    """
    if pool.empty:
        return None, {"reason": "no_history"}

    window = pool[
        (pool["target"] == target)
        & (pool["forecast_date"] < pd.Timestamp(target_date))
        & (pool["forecast_date"] >= pd.Timestamp(target_date) - pd.Timedelta(days=window_days))
    ]
    if window.empty:
        return None, {"reason": "no_history"}

    stats = window.groupby("model")["abs_error"].agg(mae="mean", n="count")
    eligible = stats[stats["n"] >= min_days]
    if eligible.empty:
        return None, {"reason": "no_eligible_candidate", "candidate_days": stats["n"].to_dict()}

    best_model = eligible["mae"].idxmin()
    return best_model, {"mae": float(eligible.loc[best_model, "mae"]), "n": int(eligible.loc[best_model, "n"])}


def rank_eligible_models(
    pool: pd.DataFrame,
    target: str,
    target_date: date,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_days: int = DEFAULT_MIN_DAYS,
) -> list[str]:
    """Every eligible candidate for `target` as of target_date, best (lowest
    MAE) first - same eligibility rule as select_best_model, but the full
    order rather than just the winner.

    _predictions_for_date walks this list and falls through to the next-
    ranked candidate whenever the current one has no actual value for this
    specific date - a real, if rare, case: a source that's been the most
    accurate all month can still have a one-off collection gap on the exact
    day being predicted (confirmed in practice - Melbourne's top wind/cloud/
    pressure/precipitation performer was missing its own forecast row for a
    single date, leaving those targets unfilled even though a perfectly good
    second-best candidate had a value that day). Without this fallback,
    "the single best candidate happened to be silent today" reads as a gap
    in the chart, even though better information was available.
    """
    if pool.empty:
        return []
    window = pool[
        (pool["target"] == target)
        & (pool["forecast_date"] < pd.Timestamp(target_date))
        & (pool["forecast_date"] >= pd.Timestamp(target_date) - pd.Timedelta(days=window_days))
    ]
    if window.empty:
        return []
    stats = window.groupby("model")["abs_error"].agg(mae="mean", n="count")
    eligible = stats[stats["n"] >= min_days].sort_values("mae")
    return list(eligible.index)


def _latest_row(conn, table: str, location: Location, target_date: date):
    return conn.execute(
        f"SELECT * FROM {table} WHERE location_name = ? AND forecast_date = ? ORDER BY generated_at DESC LIMIT 1",
        (location.name, target_date.isoformat()),
    ).fetchone()


def _candidate_value_live(db_path: Path, location: Location, model: str, target: str, target_date: date) -> float | None:
    """The already-computed forecast value a specific candidate produced for
    target_date - which has no actual yet, so it can't be read back out of
    the scored long_df the way a historical backcast can. did_rain always
    comes back as a 0-1 probability fraction, whichever candidate wins.
    """
    with db.connect(db_path) as conn:
        if model == MODEL_ENSEMBLE:
            row = _latest_row(conn, "ensemble_predictions", location, target_date)
            if row is None:
                return None
            if target == "did_rain":
                return row["rain_probability"] / 100.0 if row["rain_probability"] is not None else None
            return row[target]
        if model == MODEL_ML:
            row = _latest_row(conn, "ml_predictions", location, target_date)
            if row is None:
                return None
            if target == "did_rain":
                return row["did_rain_probability"]
            return row[target]

    # Any other candidate is a raw source's own forecast row.
    forecasts = latest_forecasts_for_date(db_path, location, target_date)
    source_row = forecasts[forecasts["source"] == model]
    if source_row.empty:
        return None
    if target == "did_rain":
        value = source_row["rain_probability"].iloc[0]
        return None if pd.isna(value) else float(value) / 100.0
    value = source_row[target].iloc[0]
    return None if pd.isna(value) else float(value)


def _insert_best_prediction(
    conn, location: Location, forecast_date: str, generated_at: str, window_days: int, min_days: int, predictions: dict[str, Any]
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO best_predictions (
            location_name, lat, lon, forecast_date, generated_at, window_days, min_days,
            max_temp, min_temp, precipitation_sum, did_rain_probability,
            wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location.name, location.lat, location.lon, forecast_date, generated_at, window_days, min_days,
            predictions.get("max_temp"), predictions.get("min_temp"), predictions.get("precipitation_sum"),
            predictions.get("did_rain_probability"),
            predictions.get("wind_speed"), predictions.get("wind_gusts"),
            predictions.get("cloud_cover"), predictions.get("humidity"), predictions.get("pressure_msl"),
            None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
        ),
    )


def _period_values_for_model(periods_conn, location: Location, model: str, target_date: date) -> dict[str, float | None]:
    """The winning daily-precipitation candidate's own 4 period forecasts for
    target_date - a plain "what did this model forecast" lookup, not a scored
    one, so unlike _candidate_value_live this is the same query whether
    target_date is in the past (a backcast) or "tomorrow" (live): neither
    case needs an actual to already exist for it.
    """
    values: dict[str, float | None] = {}
    d_iso = target_date.isoformat()
    for period in PERIODS:
        if model == MODEL_ENSEMBLE:
            row = periods_conn.execute(
                "SELECT precipitation_sum FROM ensemble_predictions_periods "
                "WHERE location_name = ? AND forecast_date = ? AND period = ? ORDER BY generated_at DESC LIMIT 1",
                (location.name, d_iso, period),
            ).fetchone()
        elif model == MODEL_ML:
            row = periods_conn.execute(
                "SELECT precipitation_sum FROM ml_predictions_periods "
                "WHERE location_name = ? AND forecast_date = ? AND period = ? ORDER BY generated_at DESC LIMIT 1",
                (location.name, d_iso, period),
            ).fetchone()
        else:
            row = periods_conn.execute(
                "SELECT precipitation_sum FROM forecast_periods "
                "WHERE location_name = ? AND forecast_date = ? AND period = ? AND source = ? "
                "ORDER BY CASE WHEN collection_method = 'live' THEN 0 ELSE 1 END, collected_at DESC LIMIT 1",
                (location.name, d_iso, period, model),
            ).fetchone()
        values[period] = row["precipitation_sum"] if row is not None else None
    return values


def _insert_best_period_predictions(
    periods_conn,
    location: Location,
    forecast_date: str,
    generated_at: str,
    window_days: int,
    min_days: int,
    period_values: dict[str, float | None],
) -> None:
    for period, value in period_values.items():
        if value is None:
            continue
        periods_conn.execute(
            """
            INSERT OR IGNORE INTO best_predictions_periods (
                location_name, lat, lon, forecast_date, period, generated_at, window_days, min_days,
                precipitation_sum, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name, location.lat, location.lon, forecast_date, period, generated_at, window_days, min_days,
                value,
                None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
            ),
        )


def _select_precip_and_rain(
    pool: pd.DataFrame,
    target_date: date,
    window_days: int,
    min_days: int,
    value_for: Callable[[str, str], float | None],
    periods_for: Callable[[str], dict[str, float | None]],
) -> tuple[dict[str, Any], dict[str, str], dict[str, float | None]]:
    """precipitation_sum and did_rain are chosen together, from whichever
    PERIOD_AWARE_CANDIDATES model has the lowest trailing MAE on
    precipitation_sum specifically - unlike every other BEST_TARGETS entry,
    did_rain is NOT ranked independently. This guarantees the winner is
    always a source that reports real period data in general, and that
    Best's rain probability, precipitation total, and every period figure all
    come from the exact same source instead of three independently-selected
    winners that could disagree with each other.

    Among the ranked candidates, the first one that ALSO actually has period
    data for this exact target_date wins outright - not just the first one
    with a resolvable precipitation_sum value. A period-aware source can
    still have a one-off gap in its own period collection on one specific
    date (e.g. a brand-new live-only source whose period tracking came
    online a day after its daily forecasts did - confirmed in practice), and
    without this check that source could still win the daily figure while
    leaving every period blank, exactly the kind of gap this whole selection
    scheme exists to avoid. Only if NO ranked candidate has any period data
    at all for this date does the search fall back to the best candidate
    with just a value (the same graceful-degradation rule every other target
    already gets, see rank_eligible_models) - Best still reports its best
    available precipitation/rain figure rather than losing the day entirely,
    with periods left blank as a genuine last resort.
    """
    predictions: dict[str, Any] = {}
    chosen: dict[str, str] = {}
    period_values: dict[str, float | None] = {}

    restricted = pool if pool.empty else pool[pool["model"].isin(PERIOD_AWARE_CANDIDATES)]
    precip_model = None
    precip_value = None
    fallback_model = None
    fallback_value = None
    fallback_periods: dict[str, float | None] = {}

    for candidate in rank_eligible_models(restricted, "precipitation_sum", target_date, window_days, min_days):
        value = value_for(candidate, "precipitation_sum")
        if value is None:
            continue
        candidate_periods = periods_for(candidate)
        if fallback_model is None:
            fallback_model, fallback_value, fallback_periods = candidate, value, candidate_periods
        if any(v is not None for v in candidate_periods.values()):
            precip_model, precip_value, period_values = candidate, value, candidate_periods
            break

    if precip_model is None:
        if fallback_model is None:
            return predictions, chosen, period_values
        precip_model, precip_value, period_values = fallback_model, fallback_value, fallback_periods

    predictions["precipitation_sum"] = round(float(precip_value), 2)
    chosen["precipitation_sum"] = precip_model

    rain_value = value_for(precip_model, "did_rain")
    if rain_value is not None:
        predictions["did_rain_probability"] = round(float(rain_value), 3)
        chosen["did_rain"] = precip_model

    return predictions, chosen, period_values


def _predictions_for_date(
    pool: pd.DataFrame,
    target_date: date,
    window_days: int,
    min_days: int,
    value_for: Callable[[str, str], float | None],
    periods_for: Callable[[str], dict[str, float | None]],
) -> tuple[dict[str, Any], dict[str, str], dict[str, float | None]]:
    """Shared per-target selection loop: for every BEST_TARGETS entry except
    precipitation_sum/did_rain (handled together first, see
    _select_precip_and_rain), walk rank_eligible_models's order (best MAE
    first) and use the first candidate whose value actually resolves via
    `value_for(model, target)` - not just the single top-ranked one, since
    even a consistently-accurate candidate can have a one-off gap on this
    exact date (see rank_eligible_models). The live and backcast paths differ
    only in how that lookup is done (a DB query for "tomorrow", vs. reading
    the already-scored `pool` for a past date).
    """
    predictions, chosen, period_values = _select_precip_and_rain(
        pool, target_date, window_days, min_days, value_for, periods_for
    )
    for target in BEST_TARGETS:
        if target in ("precipitation_sum", "did_rain"):
            continue
        value = None
        model = None
        for candidate in rank_eligible_models(pool, target, target_date, window_days, min_days):
            value = value_for(candidate, target)
            if value is not None:
                model = candidate
                break
        if model is None:
            continue
        predictions[target] = round(float(value), 2)
        chosen[target] = model
    return predictions, chosen, period_values


def predict_best(
    db_path: Path,
    location: Location,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_days: int = DEFAULT_MIN_DAYS,
    target_date: date | None = None,
) -> dict[str, Any]:
    """Generate tomorrow's (or target_date's) adaptive 'Best' prediction.

    For each target independently: find whichever candidate has the lowest
    trailing window_days-day MAE in this location (min_days eligibility rule
    - see select_best_model), then copy THAT candidate's own already-computed
    forecast for target_date. "Best" never predicts anything itself; it only
    ever selects among predictions that already exist.
    """
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)

    # build_predictions_long's window_days cutoff is relative to *today*, not
    # target_date - normally the same thing (target_date defaults to
    # tomorrow), but reconstructing a specific past target_date (as
    # blend_forecast/predict_latest_ml also support) needs the fetch to
    # reach back window_days+5 (a small buffer) from target_date itself, not
    # from today. Translated into build_predictions_long's own "days back
    # from today" terms: however many days today already is past target_date,
    # plus that same window_days+5.
    lookback_days = (local_today(location) - target_date).days + window_days + 5
    long_df = build_predictions_long(db_path, [location], window_days=max(lookback_days, window_days + 5))
    pool = candidate_pool(long_df)
    if not pool.empty:
        pool = pool.copy()
        pool["forecast_date"] = pd.to_datetime(pool["forecast_date"])

    def value_for(model: str, target: str) -> float | None:
        return _candidate_value_live(db_path, location, model, target, target_date)

    with db.connect_periods(get_periods_db_path(db_path)) as periods_conn:

        def periods_for(model: str) -> dict[str, float | None]:
            return _period_values_for_model(periods_conn, location, model, target_date)

        predictions, chosen, period_values = _predictions_for_date(
            pool, target_date, window_days, min_days, value_for, periods_for
        )
        if not predictions:
            return {"error": "No eligible candidate found for any target.", "forecast_date": target_date.isoformat()}

        generated_at = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
        with db.connect(db_path) as conn:
            _insert_best_prediction(conn, location, target_date.isoformat(), generated_at, window_days, min_days, predictions)
            conn.commit()

        if "precipitation_sum" in chosen and any(v is not None for v in period_values.values()):
            _insert_best_period_predictions(
                periods_conn, location, target_date.isoformat(), generated_at, window_days, min_days, period_values
            )
            periods_conn.commit()

    return {"forecast_date": target_date.isoformat(), "predictions": predictions, "chosen_sources": chosen}
