from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from weather_ensemble import db
from weather_ensemble.config import FORECAST_VARIABLES, TARGETS, Location, get_periods_db_path
from weather_ensemble.service import (
    default_forecast_target_date,
    latest_forecast_periods_for_date,
    latest_forecasts_for_date,
    load_modelling_table,
    load_period_modelling_table,
)

TARGET_MAP = {target: f"actual_{target}" for target in TARGETS}
CLASSIFICATION_TARGETS = {"did_rain"}
MODEL_VERSION = "phase4-ridge-logreg-narrow-features-v1"

DATE_FEATURES = ["month", "day_of_year", "day_of_week"]

# did_rain has no forecast variable of its own (it's derived from precipitation),
# so its features are drawn from precipitation_sum's columns instead.
FEATURE_TARGET_OVERRIDE = {"did_rain": "precipitation_sum"}

# Ridge has no non-negativity constraint, so it can predict a slightly
# negative value for a quantity that physically can't go below zero - most
# visibly precipitation on a dry day. These are clipped at 0 wherever a
# regression prediction is produced or evaluated, so a model's reported MAE
# and its served prediction agree.
NON_NEGATIVE_TARGETS = {
    "precipitation_sum", "wind_speed", "wind_gusts",
    "cloud_cover", "humidity", "pressure_msl",
}

# Ridge has no upper bound either. cloud_cover/humidity are percentages capped
# at 100: a row with missing/imputed features can push the model past it.
# precipitation_sum is capped at 500mm because Ridge can extrapolate wildly on
# the rare heavy-rain rows least represented in training; Australia's 24h
# rainfall record is ~907mm, so 500mm bounds any plausible single-day forecast
# while still catching runaway extrapolation. Wind has no comparable ceiling.
#
# Precipitation is fitted with plain Ridge plus this ceiling, not a
# log1p/expm1 target transform: expm1's exponential inversion turns a
# moderately-high log-space prediction into an extreme one, and that approach
# scored markedly worse in backtests.
CEILING_TARGETS = {"cloud_cover": 100.0, "humidity": 100.0, "precipitation_sum": 500.0}


def clip_prediction(target_name: str, value: float) -> float:
    """Clamp a regression prediction to [0, CEILING_TARGETS] for NON_NEGATIVE_TARGETS."""
    if target_name not in NON_NEGATIVE_TARGETS:
        return value
    value = max(0.0, value)
    ceiling = CEILING_TARGETS.get(target_name)
    if ceiling is not None:
        value = min(ceiling, value)
    return value


@dataclass(frozen=True)
class TrainedModelBundle:
    target: str
    features: list[str]
    model: Pipeline
    metrics: dict[str, float]
    trained_at: str
    model_type: str


def _build_wide_feature_table(long_df: pd.DataFrame, include_targets: bool) -> pd.DataFrame:
    """Build wide source-agreement features from a long forecast table.

    One row per (location_name, forecast_date): a `{source}__{var}` column per
    source and FORECAST_VARIABLES entry, month/day_of_year/day_of_week, and
    per-variable cross-source mean/median/std/min/max/range/count columns.
    """
    if long_df.empty:
        return long_df

    base_cols = ["location_name", "forecast_date"]
    if include_targets:
        base_cols.extend(TARGET_MAP.values())
    base_cols = [c for c in base_cols if c in long_df.columns]
    base = long_df[base_cols].drop_duplicates(subset=["location_name", "forecast_date"])

    frames = [base.set_index(["location_name", "forecast_date"])]
    for var in FORECAST_VARIABLES:
        if var not in long_df.columns:
            continue
        pivot = long_df.pivot_table(
            index=["location_name", "forecast_date"],
            columns="source",
            values=var,
            aggfunc="last",
        )
        pivot.columns = [f"{source}__{var}" for source in pivot.columns]
        frames.append(pivot)

    wide = pd.concat(frames, axis=1).reset_index()
    wide["forecast_date"] = pd.to_datetime(wide["forecast_date"])
    wide["month"] = wide["forecast_date"].dt.month
    wide["day_of_year"] = wide["forecast_date"].dt.dayofyear
    wide["day_of_week"] = wide["forecast_date"].dt.dayofweek

    for var in FORECAST_VARIABLES:
        source_cols = [c for c in wide.columns if c.endswith(f"__{var}")]
        if source_cols:
            wide[f"source_mean__{var}"] = wide[source_cols].mean(axis=1)
            wide[f"source_median__{var}"] = wide[source_cols].median(axis=1)
            wide[f"source_std__{var}"] = wide[source_cols].std(axis=1, ddof=0)
            wide[f"source_min__{var}"] = wide[source_cols].min(axis=1)
            wide[f"source_max__{var}"] = wide[source_cols].max(axis=1)
            wide[f"source_range__{var}"] = wide[f"source_max__{var}"] - wide[f"source_min__{var}"]
            wide[f"source_count__{var}"] = wide[source_cols].notna().sum(axis=1)

            # A source missing on one day (a collection gap, a provider
            # outage) is filled from this row's own cross-source median, not
            # left to the pipeline's SimpleImputer, whose column-wide training
            # median has no relationship to this day's conditions and can push
            # Ridge into physically impossible predictions. The fill happens
            # after the spread stats above, so source_std/source_range reflect
            # only the sources that actually reported.
            wide[source_cols] = wide[source_cols].apply(lambda col, var=var: col.fillna(wide[f"source_median__{var}"]))

    return wide.sort_values("forecast_date")


def build_feature_table(db_path: Path, location: Location, window_days: int | None = None) -> pd.DataFrame:
    """Build one wide training row per forecast date that has an actual.

    Includes source-specific forecast columns plus ensemble statistics across
    sources: mean, median, std, min, max, range and count. These
    disagreement/spread features are often as important as the raw provider
    values.

    window_days keeps only forecast dates within that many days of the
    location's local today (None uses all history). A bounded window keeps
    training representative of the sources live today: unbounded history
    leaves recently added sources mostly null and keeps fitting retired
    sources and superseded provider models.
    """
    long_df = load_modelling_table(db_path, location, window_days=window_days)
    return _build_wide_feature_table(long_df, include_targets=True)


def build_prediction_feature_table(db_path: Path, location: Location, target_date: date | None = None) -> pd.DataFrame:
    """Build a wide feature row for target_date (default: default_forecast_target_date).

    Uses each source's most recently collected forecast for that date.
    """
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)
    long_df = latest_forecasts_for_date(db_path, location, target_date)
    return _build_wide_feature_table(long_df, include_targets=False)


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"location_name", "forecast_date", *TARGET_MAP.values()}
    # Keep numeric features only; the model pipelines cannot consume strings/datetimes.
    cols = [c for c in df.columns if c not in excluded]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def features_for_target(df: pd.DataFrame, var: str) -> list[str]:
    """Only this variable's own signals: its per-source forecasts, cross-source stats, and date features.

    Using every forecast variable's columns to predict every target is a p >> n setup on a dataset
    this small, and overfits badly (see notebooks/ml_model_comparison.ipynb).
    source_count is excluded: it is close to constant across rows, so it carries little signal.
    """
    cols = [c for c in df.columns if c.endswith(f"__{var}") and c != f"source_count__{var}"]
    return cols + [c for c in DATE_FEATURES if c in df.columns]


def _make_model(target_name: str) -> tuple[str, Pipeline]:
    if target_name in CLASSIFICATION_TARGETS:
        return (
            "classification",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LogisticRegression(max_iter=1000)),
                ]
            ),
        )
    return (
        "regression",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
    )


def train_models(
    db_path: Path,
    location: Location,
    output_dir: Path,
    min_rows: int = 30,
    test_size: float = 0.25,
    window_days: int | None = None,
) -> dict[str, Any]:
    """Train one model per TARGETS entry for this location and pickle it to output_dir.

    did_rain is a LogisticRegression classifier; every other target is Ridge
    (alpha=1.0), both behind a median SimpleImputer. Rows are split
    chronologically (no shuffle): the oldest (1 - test_size) train the saved
    model and the newest test_size are held out for the reported metrics.
    A target is skipped with fewer than min_rows labelled rows. Writes
    `{target}.pkl` (a TrainedModelBundle) per target plus training_summary.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_feature_table(db_path, location, window_days=window_days)
    if df.empty:
        return {"error": "No modelling rows available. Run backfill first."}

    results: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "location": location.name,
        "rows_available": len(df),
        "trained_at": datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds"),
        "targets": {},
    }

    for target_name, target_col in TARGET_MAP.items():
        if target_col not in df.columns:
            results["targets"][target_name] = {"status": "skipped", "reason": f"Missing {target_col}."}
            continue

        feature_var = FEATURE_TARGET_OVERRIDE.get(target_name, target_name)
        features = features_for_target(df, feature_var)
        if not features:
            results["targets"][target_name] = {"status": "skipped", "reason": f"No feature columns available for '{feature_var}'."}
            continue

        data = df[features + [target_col]].dropna(subset=[target_col])
        if len(data) < min_rows:
            results["targets"][target_name] = {"status": "skipped", "reason": f"Need at least {min_rows} rows, only found {len(data)}."}
            continue

        X = data[features]
        y = data[target_col]
        if target_name in CLASSIFICATION_TARGETS and y.nunique() < 2:
            results["targets"][target_name] = {"status": "skipped", "reason": "Classification target has only one class."}
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        model_type, model = _make_model(target_name)
        model.fit(X_train, y_train)

        if model_type == "classification":
            preds = model.predict(X_test)
            metrics = {"accuracy": float(accuracy_score(y_test, preds)), "train_rows": len(X_train), "test_rows": len(X_test)}
            try:
                probs = model.predict_proba(X_test)[:, 1]
                metrics["auc"] = float(roc_auc_score(y_test, probs))
            except ValueError:
                # y_test can end up with only one class present (a small/
                # imbalanced split) - roc_auc_score requires both classes, so
                # AUC is omitted for that target rather than failing training.
                metrics["auc"] = None
        else:
            preds = [clip_prediction(target_name, p) for p in model.predict(X_test)]
            metrics = {
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }

        bundle = TrainedModelBundle(target=target_name, features=features, model=model, metrics=metrics, trained_at=results["trained_at"], model_type=model_type)
        with (output_dir / f"{target_name}.pkl").open("wb") as f:
            pickle.dump(bundle, f)
        results["targets"][target_name] = {
            "status": "trained",
            "model_type": model_type,
            "feature_count": len(features),
            **metrics,
        }

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def load_model_bundle(model_dir: Path, target: str) -> TrainedModelBundle:
    with (model_dir / f"{target}.pkl").open("rb") as f:
        return pickle.load(f)


def build_period_feature_table(
    db_path: Path, location: Location, period: str, window_days: int | None = None
) -> pd.DataFrame:
    """Sub-daily rain: mirrors build_feature_table, one period at a time.

    _build_wide_feature_table skips any FORECAST_VARIABLES entry not present
    in the input (period rows carry only precipitation_sum and
    rain_probability), and its include_targets base columns pick up
    actual_precipitation_sum/actual_did_rain from the period actuals.
    """
    long_df = load_period_modelling_table(db_path, location, period, window_days=window_days)
    return _build_wide_feature_table(long_df, include_targets=True)


def build_period_prediction_feature_table(
    db_path: Path, location: Location, period: str, target_date: date | None = None
) -> pd.DataFrame:
    """Mirrors build_prediction_feature_table, one period at a time."""
    if target_date is None:
        target_date = default_forecast_target_date(db_path, location)
    long_df = latest_forecast_periods_for_date(db_path, location, period, target_date)
    return _build_wide_feature_table(long_df, include_targets=False)


def train_period_model(
    db_path: Path,
    location: Location,
    period: str,
    output_dir: Path,
    min_rows: int = 30,
    test_size: float = 0.25,
    window_days: int | None = None,
) -> dict[str, Any]:
    """Sub-daily rain: a single precipitation_sum Ridge model per (location,
    period), trained and evaluated like train_models' regression targets and
    saved as `precipitation_sum_{period}.pkl`. Period models predict
    precipitation_sum only; there is no did_rain classifier per period.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_period_feature_table(db_path, location, period, window_days=window_days)
    if df.empty:
        return {"error": "No modelling rows available. Run --backfill first.", "period": period}

    target_col = "actual_precipitation_sum"
    if target_col not in df.columns:
        return {"error": f"Missing {target_col}.", "period": period}

    features = features_for_target(df, "precipitation_sum")
    if not features:
        return {"error": "No feature columns available for 'precipitation_sum'.", "period": period}

    data = df[features + [target_col]].dropna(subset=[target_col])
    if len(data) < min_rows:
        return {"error": f"Need at least {min_rows} rows, only found {len(data)}.", "period": period}

    X, y = data[features], data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model_type, model = _make_model("precipitation_sum")
    model.fit(X_train, y_train)

    preds = [clip_prediction("precipitation_sum", p) for p in model.predict(X_test)]
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }
    trained_at = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")

    bundle = TrainedModelBundle(
        target="precipitation_sum", features=features, model=model,
        metrics=metrics, trained_at=trained_at, model_type=model_type,
    )
    with (output_dir / f"precipitation_sum_{period}.pkl").open("wb") as f:
        pickle.dump(bundle, f)

    return {
        "model_version": MODEL_VERSION,
        "location": location.name,
        "period": period,
        "rows_available": len(df),
        "trained_at": trained_at,
        "status": "trained",
        "model_type": model_type,
        "feature_count": len(features),
        **metrics,
    }


def predict_latest_ml_period(
    db_path: Path, location: Location, period: str, model_dir: Path, target_date: date | None = None
) -> dict[str, Any]:
    """Mirrors predict_latest_ml, narrowed to the single precipitation_sum-per-period model."""
    df = build_period_prediction_feature_table(db_path, location, period, target_date=target_date)
    if df.empty:
        return {"error": "No forecast periods found. Run collect_forecast_periods first.", "period": period}

    model_path = model_dir / f"precipitation_sum_{period}.pkl"
    if not model_path.exists():
        return {"error": "No trained model found. Run train_period_model first.", "period": period}

    latest = df.sort_values("forecast_date").tail(1)
    forecast_date = latest["forecast_date"].iloc[0].date().isoformat()

    with model_path.open("rb") as f:
        bundle: TrainedModelBundle = pickle.load(f)

    X = latest.reindex(columns=bundle.features)
    X = _fill_missing_sources_from_row_median(X, latest, "precipitation_sum")
    prediction = round(clip_prediction("precipitation_sum", float(bundle.model.predict(X)[0])), 2)
    metadata = {"precipitation_sum": {"model_type": bundle.model_type, **bundle.metrics}}

    generated_at = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ml_predictions_periods (
                location_name, lat, lon, forecast_date, period, generated_at, model_version,
                precipitation_sum, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name, location.lat, location.lon, forecast_date, period,
                generated_at, MODEL_VERSION, prediction,
                None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
            ),
        )
        conn.commit()

    return {
        "forecast_date": forecast_date, "period": period, "model_version": MODEL_VERSION,
        "precipitation_sum": prediction, "metadata": metadata,
    }


def _fill_missing_sources_from_row_median(X: pd.DataFrame, raw: pd.DataFrame, var: str) -> pd.DataFrame:
    """Fill any of var's per-source feature columns that are missing in X after
    reindexing to the trained model's feature set, using this row's own
    cross-source median for var - computed from whatever sources actually
    reported in `raw`, not the historical training median the pipeline's
    SimpleImputer would otherwise use.

    _build_wide_feature_table's fill only covers a source whose column exists
    in the table being built; a source absent from a single-day live
    prediction has no column there at all (reindex introduces it as all-NaN),
    so this covers that case.
    """
    source_cols = [c for c in X.columns if c.endswith(f"__{var}") and not c.startswith("source_")]
    if not source_cols:
        return X
    row_median = raw.reindex(columns=source_cols).median(axis=1)
    X = X.copy()
    X[source_cols] = X[source_cols].apply(lambda col: col.fillna(row_median))
    return X


def predict_latest_ml(db_path: Path, location: Location, model_dir: Path, target_date: date | None = None) -> dict[str, Any]:
    """Predict target_date with the currently trained models and store it in ml_predictions.

    target_date defaults to default_forecast_target_date (normally the
    location's local tomorrow). Passing it explicitly lets a run use one
    resolved date for every step, or reconstruct a specific past day's
    prediction.
    """
    df = build_prediction_feature_table(db_path, location, target_date=target_date)
    if df.empty:
        return {"error": "No forecasts found. Run collect first."}

    latest = df.sort_values("forecast_date").tail(1)
    forecast_date = latest["forecast_date"].iloc[0].date().isoformat()

    predictions: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for target in TARGET_MAP:
        path = model_dir / f"{target}.pkl"
        if not path.exists():
            continue
        bundle = load_model_bundle(model_dir, target)
        X = latest.reindex(columns=bundle.features)
        feature_var = FEATURE_TARGET_OVERRIDE.get(target, target)
        X = _fill_missing_sources_from_row_median(X, latest, feature_var)
        if bundle.model_type == "classification":
            klass = int(bundle.model.predict(X)[0])
            predictions[target] = klass
            if hasattr(bundle.model, "predict_proba"):
                predictions[f"{target}_probability"] = round(float(bundle.model.predict_proba(X)[0][1]), 3)
        else:
            predictions[target] = round(clip_prediction(target, float(bundle.model.predict(X)[0])), 2)
        metadata[target] = {"model_type": bundle.model_type, **bundle.metrics}

    if not predictions:
        return {"error": "No trained models found. Run --train first."}

    generated_at = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
    with db.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO ml_predictions (
                location_name, lat, lon, forecast_date, generated_at, model_version,
                max_temp, min_temp, precipitation_sum, did_rain, did_rain_probability,
                wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                location.name, location.lat, location.lon, forecast_date, generated_at, MODEL_VERSION,
                predictions.get("max_temp"), predictions.get("min_temp"), predictions.get("precipitation_sum"),
                predictions.get("did_rain"), predictions.get("did_rain_probability"),
                predictions.get("wind_speed"), predictions.get("wind_gusts"),
                predictions.get("cloud_cover"), predictions.get("humidity"), predictions.get("pressure_msl"),
                None,  # metadata_json: not persisted (write-only, never read back) - see archive.py
            ),
        )
        conn.commit()

    return {"forecast_date": forecast_date, "model_version": MODEL_VERSION, "predictions": predictions, "metadata": metadata}
