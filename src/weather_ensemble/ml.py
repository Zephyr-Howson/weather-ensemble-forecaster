from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from weather_ensemble.config import FORECAST_VARIABLES, Location, TARGETS
from weather_ensemble.service import load_modelling_table

TARGET_MAP = {target: f"actual_{target}" for target in TARGETS}
CLASSIFICATION_TARGETS = {"did_rain"}
MODEL_VERSION = "phase3-rf-expanded-weather-v2"


@dataclass(frozen=True)
class TrainedModelBundle:
    target: str
    features: list[str]
    model: Pipeline
    metrics: dict[str, float]
    trained_at: str
    model_type: str


def build_feature_table(db_path: Path, location: Location) -> pd.DataFrame:
    """Build one wide training row per forecast date.

    Includes source-specific forecast columns plus ensemble statistics across
    sources: mean, median, std, min, max and range. These disagreement/spread
    features are often as important as the raw provider values.
    """
    long_df = load_modelling_table(db_path, location)
    if long_df.empty:
        return long_df

    base_cols = ["location_name", "forecast_date"] + list(TARGET_MAP.values())
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

    return wide.sort_values("forecast_date")


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"location_name", "forecast_date", *TARGET_MAP.values()}
    # Keep numeric features only; RandomForest cannot consume strings/datetimes.
    cols = [c for c in df.columns if c not in excluded]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _make_model(target_name: str) -> tuple[str, Pipeline]:
    if target_name in CLASSIFICATION_TARGETS:
        return (
            "classification",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)),
                ]
            ),
        )
    return (
        "regression",
        Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)),
            ]
        ),
    )


def train_models(db_path: Path, location: Location, output_dir: Path, min_rows: int = 30, test_size: float = 0.25) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_feature_table(db_path, location)
    if df.empty:
        return {"error": "No modelling rows available. Run backfill first."}

    features = feature_columns(df)
    if not features:
        return {"error": "No numeric feature columns available."}

    results: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "location": location.name,
        "rows_available": int(len(df)),
        "feature_count": len(features),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "targets": {},
    }

    for target_name, target_col in TARGET_MAP.items():
        if target_col not in df.columns:
            results["targets"][target_name] = {"status": "skipped", "reason": f"Missing {target_col}."}
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
            metrics = {"accuracy": float(accuracy_score(y_test, preds)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test))}
            try:
                probs = model.predict_proba(X_test)[:, 1]
                metrics["auc"] = float(roc_auc_score(y_test, probs))
            except Exception:
                pass
        else:
            preds = model.predict(X_test)
            metrics = {
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
            }

        bundle = TrainedModelBundle(target=target_name, features=features, model=model, metrics=metrics, trained_at=results["trained_at"], model_type=model_type)
        with (output_dir / f"{target_name}.pkl").open("wb") as f:
            pickle.dump(bundle, f)
        results["targets"][target_name] = {"status": "trained", "model_type": model_type, **metrics}

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def load_model_bundle(model_dir: Path, target: str) -> TrainedModelBundle:
    with (model_dir / f"{target}.pkl").open("rb") as f:
        return pickle.load(f)


def predict_latest_ml(db_path: Path, location: Location, model_dir: Path) -> dict[str, Any]:
    df = build_feature_table(db_path, location)
    if df.empty:
        return {"error": "No feature rows available."}

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
        if bundle.model_type == "classification":
            klass = int(bundle.model.predict(X)[0])
            predictions[target] = klass
            if hasattr(bundle.model, "predict_proba"):
                predictions[f"{target}_probability"] = round(float(bundle.model.predict_proba(X)[0][1]), 3)
        else:
            predictions[target] = round(float(bundle.model.predict(X)[0]), 2)
        metadata[target] = {"model_type": bundle.model_type, **bundle.metrics}

    if not predictions:
        return {"error": "No trained models found. Run --train first."}

    return {"forecast_date": forecast_date, "model_version": MODEL_VERSION, "predictions": predictions, "metadata": metadata}
