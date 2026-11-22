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

# "Best" picks, per target, whichever candidate - a raw source, the Weighted
# blend, or the ML model - had the lowest MAE for that one target in this
# location recently. The two naive baselines (persistence/climatology) are
# not candidates: they exist to show how much better real forecasting is than
# doing nothing, not as strategies Best should adopt. MODEL_BEST is excluded
# too: build_predictions_long includes Best's own scored history (for the
# report's leaderboard/trend charts), and Best must not select itself - it
# has no forecast value of its own to copy.
_EXCLUDED_CANDIDATES = {BASELINE_PERSISTENCE, BASELINE_CLIMATOLOGY, MODEL_BEST}

DEFAULT_WINDOW_DAYS = 30
DEFAULT_MIN_DAYS = 14

# The daily TARGETS set. Every target except precipitation_sum/did_rain is
# selected independently (see _predictions_for_date). Sub-daily rain periods
# are not a separate selection: whichever candidate wins the daily
# precipitation_sum target has its own period breakdown copied too (see
# _period_values_for_model), so the four periods come from the same candidate
# as Best's daily total.
BEST_TARGETS: list[str] = list(TARGETS)

# Only candidates that report sub-daily rain periods: the 4 raw Open-Meteo
# models (collect_forecast_periods in service.py collects periods from
# Open-Meteo only), plus Weighted and ML, whose period predictions are built
# from those Open-Meteo period forecasts. precipitation_sum and did_rain are
# both selected from this restricted pool (see _select_precip_and_rain), so
# the winner can have period data to copy. The non-Open-Meteo providers (BOM,
# WeatherAPI, OpenWeatherMap, Weatherbit, AccuWeather, Visual Crossing,
# wttr.in) never report periods and so never win precipitation_sum/did_rain.
PERIOD_AWARE_CANDIDATES = {MODEL_ENSEMBLE, MODEL_ML} | {f"open_meteo_{m}" for m in OPEN_METEO_MODELS}


def candidate_pool(long_df: pd.DataFrame) -> pd.DataFrame:
    """Every real forecast strategy for one location - every raw source, plus
    Weighted and ML - with the two naive baselines and Best itself filtered out.
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
    ranked candidate whenever the current one has no forecast value for this
    specific date: the most accurate candidate over the window can still have
    a one-off collection gap on the day being predicted, and the next-ranked
    candidate's value is better than leaving the target blank.
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
    """precipitation_sum and did_rain are chosen together, from the
    PERIOD_AWARE_CANDIDATES ranked by trailing MAE on precipitation_sum -
    did_rain is not ranked on its own MAE. Best's rain probability,
    precipitation total and every period figure therefore all come from the
    same candidate.

    The winner is the first ranked candidate that has both a
    precipitation_sum value and at least one period value for this exact
    target_date: a period-aware candidate can still have a one-off gap in its
    period data on a given date, and picking it would leave every period
    blank. If no ranked candidate has period data for the date, the
    first-ranked candidate with a precipitation_sum value wins and the
    periods are left blank. did_rain_probability is that same candidate's
    rain probability, omitted if it has none.
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
    """Generate target_date's adaptive 'Best' prediction (default:
    default_forecast_target_date, normally the location's local tomorrow).

    For each target, rank candidates by trailing window_days-day MAE in this
    location (min_days eligibility rule - see rank_eligible_models), then copy
    the top-ranked candidate's own already-computed forecast for target_date,
    falling through to the next-ranked one if it has no value.
    precipitation_sum and did_rain are picked together from
    PERIOD_AWARE_CANDIDATES (see _select_precip_and_rain). "Best" never
    predicts anything itself; it only selects among predictions that already
    exist. Writes best_predictions, plus best_predictions_periods when the
    precipitation winner has period values.
    """
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)

    # build_predictions_long's window_days cutoff is relative to the
    # location's local today, not target_date. The fetch must reach back
    # window_days+5 (a small buffer) from target_date itself, so a past
    # target_date adds however many days today is past it.
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
