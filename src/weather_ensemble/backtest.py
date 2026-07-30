from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import Location, get_periods_db_path
from weather_ensemble.ml import (
    CLASSIFICATION_TARGETS,
    FEATURE_TARGET_OVERRIDE,
    MODEL_VERSION,
    TARGET_MAP,
    _build_wide_feature_table,
    _make_model,
    clip_prediction,
    features_for_target,
)
from weather_ensemble.service import (
    blend_period_precipitation,
    blend_weighted,
    compute_mae_scores,
    compute_period_mae_scores,
    load_modelling_table,
    load_period_modelling_table,
)

BACKTEST_MODEL_VERSION = f"backtest-{MODEL_VERSION}"
MIN_TRAIN_ROWS = 30


def _existing_forecast_dates(conn: sqlite3.Connection, table: str, location: Location) -> set[str]:
    rows = conn.execute(f"SELECT DISTINCT forecast_date FROM {table} WHERE location_name = ?", (location.name,))
    return {r[0] for r in rows}


def _existing_forecast_dates_period(conn: sqlite3.Connection, table: str, location: Location, period: str) -> set[str]:
    rows = conn.execute(
        f"SELECT DISTINCT forecast_date FROM {table} WHERE location_name = ? AND period = ?", (location.name, period)
    )
    return {r[0] for r in rows}


def _generated_at_for(target_date) -> str:
    """A deterministic 'as if generated the evening before' timestamp.

    Deterministic (not datetime.now()) so re-running the backtest for a date
    that already has a row is a true no-op via INSERT OR IGNORE, instead of
    piling up a fresh duplicate row with a new generated_at every run.
    """
    return datetime.combine(target_date - timedelta(days=1), datetime.min.time()).replace(hour=21).isoformat(timespec="seconds")


def _backtest_ensemble(
    conn: sqlite3.Connection,
    location: Location,
    long_df: pd.DataFrame,
    target_date,
    window_days: int,
    existing: set[str],
) -> str:
    d_iso = target_date.isoformat()
    if d_iso in existing:
        return "skipped_existing"

    forecast_rows = long_df[long_df["forecast_date"] == pd.Timestamp(target_date)]
    if forecast_rows.empty:
        return "skipped_no_forecast"

    history = long_df[
        (long_df["forecast_date"] < pd.Timestamp(target_date))
        & (long_df["forecast_date"] >= pd.Timestamp(target_date) - pd.Timedelta(days=window_days))
    ]
    scores = compute_mae_scores(history)
    blended, metadata = blend_weighted(forecast_rows, scores)
    metadata["backtest"] = True

    conn.execute(
        """
        INSERT OR IGNORE INTO ensemble_predictions (
            location_name, lat, lon, forecast_date, generated_at, window_days,
            max_temp, min_temp, rain_probability, precipitation_sum, did_rain,
            wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location.name, location.lat, location.lon, d_iso, _generated_at_for(target_date), window_days,
            blended.get("max_temp"), blended.get("min_temp"), blended.get("rain_probability"),
            blended.get("precipitation_sum"), blended.get("did_rain"),
            blended.get("wind_speed"), blended.get("wind_gusts"), blended.get("cloud_cover"),
            blended.get("humidity"), blended.get("pressure_msl"),
            None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
        ),
    )
    return "written"


def _backtest_ml(
    conn: sqlite3.Connection,
    location: Location,
    wide_all: pd.DataFrame,
    target_date,
    train_window_days: int,
    existing: set[str],
) -> str:
    d_iso = target_date.isoformat()
    if d_iso in existing:
        return "skipped_existing"

    predict_row = wide_all[wide_all["forecast_date"] == pd.Timestamp(target_date)]
    if predict_row.empty:
        return "skipped_no_forecast"

    train_df = wide_all[
        (wide_all["forecast_date"] < pd.Timestamp(target_date))
        & (wide_all["forecast_date"] >= pd.Timestamp(target_date) - pd.Timedelta(days=train_window_days))
    ]

    predictions: dict[str, Any] = {}
    metadata: dict[str, Any] = {"backtest": True}
    for target_name, target_col in TARGET_MAP.items():
        if target_col not in train_df.columns:
            continue
        feature_var = FEATURE_TARGET_OVERRIDE.get(target_name, target_name)
        features = features_for_target(train_df, feature_var)
        if not features:
            continue
        data = train_df[features + [target_col]].dropna(subset=[target_col])
        if len(data) < MIN_TRAIN_ROWS:
            continue
        X, y = data[features], data[target_col]
        if target_name in CLASSIFICATION_TARGETS and y.nunique() < 2:
            continue

        model_type, model = _make_model(target_name)
        model.fit(X, y)
        X_pred = predict_row.reindex(columns=features)
        if model_type == "classification":
            predictions[target_name] = int(model.predict(X_pred)[0])
            if hasattr(model, "predict_proba"):
                predictions[f"{target_name}_probability"] = round(float(model.predict_proba(X_pred)[0][1]), 3)
        else:
            predictions[target_name] = round(clip_prediction(target_name, float(model.predict(X_pred)[0])), 2)
        metadata[target_name] = {"model_type": model_type, "train_rows": len(data)}

    if not predictions:
        return "skipped_insufficient_data"

    conn.execute(
        """
        INSERT OR IGNORE INTO ml_predictions (
            location_name, lat, lon, forecast_date, generated_at, model_version,
            max_temp, min_temp, precipitation_sum, did_rain, did_rain_probability,
            wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location.name, location.lat, location.lon, d_iso, _generated_at_for(target_date), BACKTEST_MODEL_VERSION,
            predictions.get("max_temp"), predictions.get("min_temp"), predictions.get("precipitation_sum"),
            predictions.get("did_rain"), predictions.get("did_rain_probability"),
            predictions.get("wind_speed"), predictions.get("wind_gusts"),
            predictions.get("cloud_cover"), predictions.get("humidity"), predictions.get("pressure_msl"),
            None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
        ),
    )
    return "written"


def _backtest_ensemble_period(
    conn: sqlite3.Connection,
    location: Location,
    period: str,
    long_df: pd.DataFrame,
    target_date,
    window_days: int,
    existing: set[str],
) -> str:
    d_iso = target_date.isoformat()
    if d_iso in existing:
        return "skipped_existing"

    forecast_rows = long_df[long_df["forecast_date"] == pd.Timestamp(target_date)]
    if forecast_rows.empty:
        return "skipped_no_forecast"

    history = long_df[
        (long_df["forecast_date"] < pd.Timestamp(target_date))
        & (long_df["forecast_date"] >= pd.Timestamp(target_date) - pd.Timedelta(days=window_days))
    ]
    scores = compute_period_mae_scores(history)
    blended, metadata = blend_period_precipitation(forecast_rows, scores)
    metadata["backtest"] = True

    conn.execute(
        """
        INSERT OR IGNORE INTO ensemble_predictions_periods (
            location_name, lat, lon, forecast_date, period, generated_at, window_days,
            precipitation_sum, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location.name, location.lat, location.lon, d_iso, period,
            _generated_at_for(target_date), window_days, blended,
            None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
        ),
    )
    return "written"


def _backtest_ml_period(
    conn: sqlite3.Connection,
    location: Location,
    period: str,
    wide_all: pd.DataFrame,
    target_date,
    train_window_days: int,
    existing: set[str],
) -> str:
    d_iso = target_date.isoformat()
    if d_iso in existing:
        return "skipped_existing"

    predict_row = wide_all[wide_all["forecast_date"] == pd.Timestamp(target_date)]
    if predict_row.empty:
        return "skipped_no_forecast"

    train_df = wide_all[
        (wide_all["forecast_date"] < pd.Timestamp(target_date))
        & (wide_all["forecast_date"] >= pd.Timestamp(target_date) - pd.Timedelta(days=train_window_days))
    ]

    target_col = "actual_precipitation_sum"
    if target_col not in train_df.columns:
        return "skipped_no_forecast"
    features = features_for_target(train_df, "precipitation_sum")
    if not features:
        return "skipped_no_forecast"
    data = train_df[features + [target_col]].dropna(subset=[target_col])
    if len(data) < MIN_TRAIN_ROWS:
        return "skipped_insufficient_data"

    X, y = data[features], data[target_col]
    _model_type, model = _make_model("precipitation_sum")
    model.fit(X, y)
    X_pred = predict_row.reindex(columns=features)
    prediction = round(clip_prediction("precipitation_sum", float(model.predict(X_pred)[0])), 2)

    conn.execute(
        """
        INSERT OR IGNORE INTO ml_predictions_periods (
            location_name, lat, lon, forecast_date, period, generated_at, model_version,
            precipitation_sum, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location.name, location.lat, location.lon, d_iso, period,
            _generated_at_for(target_date), BACKTEST_MODEL_VERSION, prediction,
            None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
        ),
    )
    return "written"


def backtest_period_predictions(
    db_path: Path,
    location: Location,
    period: str,
    days: int,
    ensemble_window_days: int = 30,
    train_window_days: int = 90,
) -> dict[str, Any]:
    """Small-slice sub-daily rain: mirrors backtest_predictions, one period +
    precipitation_sum only. Needed for the report's ensemble/ML leaderboard
    entries to have anything to show at all - a live "tomorrow" prediction
    has no actual yet to score against, so without walk-forward historical
    predictions like these, the report's period cards can only ever show raw
    source accuracy, never ensemble/ML (a real gap found by checking the
    freshly-built report cards, not assumed away).
    """
    long_df = load_period_modelling_table(db_path, location, period)
    if long_df.empty:
        return {"location": location.name, "period": period, "error": "No modelling rows available."}
    long_df = long_df.copy()
    long_df["forecast_date"] = pd.to_datetime(long_df["forecast_date"])

    max_date = long_df["forecast_date"].max().date()
    target_dates = sorted(max_date - timedelta(days=i) for i in range(days))

    wide_all = _build_wide_feature_table(long_df, include_targets=True)

    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        existing_ensemble = _existing_forecast_dates_period(conn, "ensemble_predictions_periods", location, period)
        existing_ml = _existing_forecast_dates_period(conn, "ml_predictions_periods", location, period)

        ensemble_results: dict[str, list[str]] = {}
        ml_results: dict[str, list[str]] = {}
        for target_date in target_dates:
            outcome = _backtest_ensemble_period(
                conn, location, period, long_df, target_date, ensemble_window_days, existing_ensemble
            )
            ensemble_results.setdefault(outcome, []).append(target_date.isoformat())

            outcome = _backtest_ml_period(conn, location, period, wide_all, target_date, train_window_days, existing_ml)
            ml_results.setdefault(outcome, []).append(target_date.isoformat())

        conn.commit()

    return {
        "location": location.name,
        "period": period,
        "date_range": [target_dates[0].isoformat(), target_dates[-1].isoformat()],
        "ensemble": {k: len(v) for k, v in ensemble_results.items()},
        "ml": {k: len(v) for k, v in ml_results.items()},
    }


def backtest_predictions(
    db_path: Path,
    location: Location,
    days: int,
    ensemble_window_days: int = 30,
    train_window_days: int = 90,
) -> dict[str, Any]:
    """Regenerate ensemble + ML predictions for each of the past `days` days, walk-forward.

    For target date D, the ensemble's MAE weighting and the ML model's training
    data only use rows with forecast_date < D - exactly as if D were the
    present, so nothing D "shouldn't know yet" leaks in. A fresh Ridge/
    LogisticRegression model is trained from scratch for every date (this is
    what makes it a true walk-forward backtest rather than one model scored
    against its own future). Dates that already have a real prediction are
    left untouched; this only fills gaps.
    """
    long_df = load_modelling_table(db_path, location)
    if long_df.empty:
        return {"location": location.name, "error": "No modelling rows available."}
    long_df = long_df.copy()
    long_df["forecast_date"] = pd.to_datetime(long_df["forecast_date"])

    max_date = long_df["forecast_date"].max().date()
    target_dates = sorted(max_date - timedelta(days=i) for i in range(days))

    wide_all = _build_wide_feature_table(long_df, include_targets=True)

    with db.connect(db_path) as conn:
        existing_ensemble = _existing_forecast_dates(conn, "ensemble_predictions", location)
        existing_ml = _existing_forecast_dates(conn, "ml_predictions", location)

        ensemble_results: dict[str, list[str]] = {}
        ml_results: dict[str, list[str]] = {}
        for target_date in target_dates:
            outcome = _backtest_ensemble(conn, location, long_df, target_date, ensemble_window_days, existing_ensemble)
            ensemble_results.setdefault(outcome, []).append(target_date.isoformat())

            outcome = _backtest_ml(conn, location, wide_all, target_date, train_window_days, existing_ml)
            ml_results.setdefault(outcome, []).append(target_date.isoformat())

        conn.commit()

    return {
        "location": location.name,
        "date_range": [target_dates[0].isoformat(), target_dates[-1].isoformat()],
        "ensemble": {k: len(v) for k, v in ensemble_results.items()},
        "ml": {k: len(v) for k, v in ml_results.items()},
    }
