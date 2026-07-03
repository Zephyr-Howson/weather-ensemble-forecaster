from __future__ import annotations

from pathlib import Path
from typing import Any

from weather_ensemble.config import Location
from weather_ensemble.ml import build_feature_table, predict_latest_ml, train_models
from weather_ensemble.service import backfill, collect_forecasts, record_actual


def deploy_all_phases(
    db_path: Path,
    location: Location,
    days_back: int,
    processed_path: Path,
    model_dir: Path,
) -> dict[str, Any]:
    """Run the full local Phase 1-3 pipeline.

    Phase 1: Backfill historical forecasts + actual observations.
    Phase 2: Build/export model-ready wide training dataset.
    Phase 3: Train ML post-processing models and produce latest prediction when possible.
    """
    backfill(db_path, location, days_back)
    collected = collect_forecasts(db_path, location)
    record_actual(db_path, location)

    feature_df = build_feature_table(db_path, location)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    if processed_path.suffix.lower() == ".csv":
        feature_df.to_csv(processed_path, index=False)
    else:
        try:
            feature_df.to_parquet(processed_path, index=False)
        except Exception:
            processed_path = processed_path.with_suffix(".csv")
            feature_df.to_csv(processed_path, index=False)

    training = train_models(db_path, location, output_dir=model_dir, min_rows=min(30, max(10, days_back // 2)))
    prediction = predict_latest_ml(db_path, location, model_dir=model_dir)

    return {
        "phase_1_collection": {
            "status": "complete",
            "days_back": days_back,
            "live_forecasts_collected": len(collected),
            "db_path": str(db_path),
        },
        "phase_2_dataset": {
            "status": "complete",
            "rows": int(len(feature_df)),
            "columns": int(len(feature_df.columns)) if not feature_df.empty else 0,
            "path": str(processed_path),
        },
        "phase_3_model": {
            "status": "complete" if "error" not in training else "incomplete",
            "training": training,
            "latest_prediction": prediction,
            "model_dir": str(model_dir),
        },
    }
