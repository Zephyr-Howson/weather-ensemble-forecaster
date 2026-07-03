from __future__ import annotations

from datetime import date, datetime, timedelta
import pickle

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from weather_ensemble.config import Location
from weather_ensemble.db import connect, insert_forecasts
from weather_ensemble.ml import TrainedModelBundle, build_prediction_feature_table, predict_latest_ml
from weather_ensemble.models import ForecastRecord


def _forecast_record(source: str, forecast_date: date, collected_at: datetime, value: float) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name="Melbourne",
        lat=-37.8136,
        lon=144.9631,
        forecast_date=forecast_date,
        collected_at=collected_at,
        max_temp=value,
        min_temp=value - 5,
        rain_probability=25.0,
        precipitation_sum=1.0,
        uv_index=3.0,
        wind_speed=20.0,
        wind_gusts=30.0,
        cloud_cover=40.0,
        humidity=50.0,
        pressure_msl=1012.0,
        weather_code=1.0,
        raw_json={"source": source},
    )


def test_build_prediction_feature_table_uses_tomorrow_forecasts(tmp_path):
    db_path = tmp_path / "weather.db"
    location = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")
    tomorrow = date.today() + timedelta(days=1)

    with connect(db_path) as conn:
        insert_forecasts(
            conn,
            [
                _forecast_record("open_meteo_best_match", tomorrow, datetime.now().replace(microsecond=0), 25.0),
                _forecast_record("open_meteo_gfs_global", tomorrow, datetime.now().replace(microsecond=0), 27.0),
            ],
        )

    df = build_prediction_feature_table(db_path, location)

    assert len(df) == 1
    assert df["forecast_date"].iloc[0].date() == tomorrow
    assert "open_meteo_best_match__max_temp" in df.columns
    assert "open_meteo_gfs_global__max_temp" in df.columns


def test_predict_latest_ml_uses_live_forecast_rows(tmp_path):
    db_path = tmp_path / "weather.db"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    location = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")
    tomorrow = date.today() + timedelta(days=1)

    with connect(db_path) as conn:
        insert_forecasts(
            conn,
            [
                _forecast_record("open_meteo_best_match", tomorrow, datetime.now().replace(microsecond=0), 25.0),
                _forecast_record("open_meteo_gfs_global", tomorrow, datetime.now().replace(microsecond=0), 27.0),
            ],
        )

    feature_df = build_prediction_feature_table(db_path, location)
    feature_columns = [column for column in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[column])]
    training_frame = pd.concat([feature_df[feature_columns], feature_df[feature_columns]], ignore_index=True)
    target = pd.Series([10.0, 12.0])

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    model.fit(training_frame, target)

    bundle = TrainedModelBundle(
        target="max_temp",
        features=feature_columns,
        model=model,
        metrics={"mae": 1.0},
        trained_at="2026-06-18T00:00:00",
        model_type="regression",
    )
    with (model_dir / "max_temp.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)

    result = predict_latest_ml(db_path, location, model_dir)

    assert result["forecast_date"] == tomorrow.isoformat()
    assert result["predictions"]["max_temp"] == 11.0
