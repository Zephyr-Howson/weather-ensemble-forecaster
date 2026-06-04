from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from weather_ensemble.config import Location, VARIABLES
from weather_ensemble.service import load_modelling_table

TARGET_MAP = {
    "max_temp": "actual_max_temp",
    "min_temp": "actual_min_temp",
    "rain_probability": "actual_rain_probability",
    "uv_index": "actual_uv_index",
    "wind_speed": "actual_wind_speed",
}

MODEL_VERSION = "phase3-random-forest-v1"


@dataclass(frozen=True)
class TrainedModelBundle:
    target: str
    features: list[str]
    model: Pipeline
    metrics: dict[str, float]
    trained_at: str


def build_feature_table(db_path: Path, location: Location) -> pd.DataFrame:
    """Build one wide training row per forecast date.

    The source-level rows stored in SQLite look like:
        date + source + forecast variables + actual variables

    Most ML models want a wide matrix instead:
        date + open_meteo_best_max_temp + open_meteo_bom_max_temp + ... + actual_max_temp
    """
    long_df = load_modelling_table(db_path, location)
    if long_df.empty:
        return long_df

    base_cols = [
        "location_name",
        "forecast_date",
        "actual_max_temp",
        "actual_min_temp",
        "actual_rain_probability",
        "actual_uv_index",
        "actual_wind_speed",
        "actual_precipitation_sum",
    ]
    base = long_df[base_cols].drop_duplicates(subset=["location_name", "forecast_date"])

    frames = [base.set_index(["location_name", "forecast_date"])]
    for var in VARIABLES:
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

    # Cross-source features: the average and disagreement between providers.
    for var in VARIABLES:
        source_cols = [c for c in wide.columns if c.endswith(f"__{var}")]
        if source_cols:
            wide[f"source_mean__{var}"] = wide[source_cols].mean(axis=1)
            wide[f"source_std__{var}"] = wide[source_cols].std(axis=1)
            wide[f"source_min__{var}"] = wide[source_cols].min(axis=1)
            wide[f"source_max__{var}"] = wide[source_cols].max(axis=1)

    wide = wide.sort_values("forecast_date")
    return wide


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "location_name",
        "forecast_date",
        "actual_max_temp",
        "actual_min_temp",
        "actual_rain_probability",
        "actual_uv_index",
        "actual_wind_speed",
        "actual_precipitation_sum",
    }
    return [c for c in df.columns if c not in excluded]


def train_models(
    db_path: Path,
    location: Location,
    output_dir: Path,
    min_rows: int = 30,
    test_size: float = 0.25,
) -> dict[str, Any]:
    """Train one regression model per weather target and save them to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_feature_table(db_path, location)
    if df.empty:
        return {"error": "No modelling rows available. Run backfill first."}

    features = feature_columns(df)
    if not features:
        return {"error": "No feature columns available."}

    results: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "location": location.name,
        "rows_available": int(len(df)),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "targets": {},
    }

    for target_name, target_col in TARGET_MAP.items():
        data = df[features + [target_col]].dropna(subset=[target_col])
        if len(data) < min_rows:
            results["targets"][target_name] = {
                "status": "skipped",
                "reason": f"Need at least {min_rows} rows, only found {len(data)}.",
            }
            continue

        X = data[features]
        y = data[target_col]

        # Keep time-order leakage low by not shuffling. Last chunk is holdout.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, preds)),
            "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        }

        bundle = TrainedModelBundle(
            target=target_name,
            features=features,
            model=model,
            metrics=metrics,
            trained_at=results["trained_at"],
        )
        with (output_dir / f"{target_name}.pkl").open("wb") as f:
            pickle.dump(bundle, f)

        results["targets"][target_name] = {"status": "trained", **metrics}

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def load_model_bundle(model_dir: Path, target: str) -> TrainedModelBundle:
    with (model_dir / f"{target}.pkl").open("rb") as f:
        return pickle.load(f)


def predict_latest_ml(
    db_path: Path,
    location: Location,
    model_dir: Path,
) -> dict[str, Any]:
    """Predict using the latest feature row that has forecasts available."""
    df = build_feature_table(db_path, location)
    if df.empty:
        return {"error": "No feature rows available."}

    latest = df.sort_values("forecast_date").tail(1)
    forecast_date = latest["forecast_date"].iloc[0].date().isoformat()

    predictions: dict[str, float] = {}
    metadata: dict[str, Any] = {}
    for target in TARGET_MAP:
        path = model_dir / f"{target}.pkl"
        if not path.exists():
            continue
        bundle = load_model_bundle(model_dir, target)
        X = latest.reindex(columns=bundle.features)
        predictions[target] = round(float(bundle.model.predict(X)[0]), 2)
        metadata[target] = bundle.metrics

    if not predictions:
        return {"error": "No trained models found. Run --train first."}

    return {
        "forecast_date": forecast_date,
        "model_version": MODEL_VERSION,
        "predictions": predictions,
        "metadata": metadata,
    }
