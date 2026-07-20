from __future__ import annotations

from datetime import date, datetime, timedelta
import pickle

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from weather_ensemble.config import Location
from weather_ensemble.db import connect, insert_forecasts, upsert_actual
from weather_ensemble.ml import (
    TrainedModelBundle,
    _fill_missing_sources_from_row_median,
    build_feature_table,
    build_prediction_feature_table,
    clip_prediction,
    predict_latest_ml,
)
from weather_ensemble.models import ActualRecord, ForecastRecord


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


def test_build_prediction_feature_table_honors_explicit_target_date(tmp_path):
    """target_date exists to reconstruct a specific past day's prediction (e.g. one
    dropped by a database reset) rather than always defaulting to tomorrow."""
    db_path = tmp_path / "weather.db"
    location = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")
    tomorrow = date.today() + timedelta(days=1)
    a_past_date = date(2026, 6, 1)

    with connect(db_path) as conn:
        insert_forecasts(
            conn,
            [
                _forecast_record("open_meteo_best_match", tomorrow, datetime.now().replace(microsecond=0), 25.0),
                _forecast_record("open_meteo_best_match", a_past_date, datetime.now().replace(microsecond=0), 18.0),
            ],
        )

    df = build_prediction_feature_table(db_path, location, target_date=a_past_date)

    assert len(df) == 1
    assert df["forecast_date"].iloc[0].date() == a_past_date
    assert df["open_meteo_best_match__max_temp"].iloc[0] == 18.0


def test_missing_source_is_filled_from_this_rows_own_median_not_left_nan(tmp_path):
    """A source missing on one specific day (a collection gap, a provider
    outage) should have its slot filled from what the OTHER sources are
    saying that same day, not a stale column-wide historical median - see
    _build_wide_feature_table's fallback fill. This is what stopped Ridge
    from extrapolating a 467.9% cloud cover prediction on a day several
    sources genuinely had a gap.

    Uses build_feature_table (the multi-day, actual-joined table backtest.py
    also builds its wide_all from) rather than build_prediction_feature_table
    (the live single-day path) - a source with no row at all on target_day
    still needs a column to exist to be "NaN, then filled" rather than simply
    absent, and that only happens once the source has reported on some other
    day within the window.
    """
    db_path = tmp_path / "weather.db"
    location = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")
    other_day = date(2026, 6, 1)
    target_day = date(2026, 6, 2)

    def _record(source: str, forecast_date: date, cloud_cover: float) -> ForecastRecord:
        return ForecastRecord(
            source=source,
            location_name=location.name,
            lat=location.lat,
            lon=location.lon,
            forecast_date=forecast_date,
            collected_at=datetime.now().replace(microsecond=0),
            cloud_cover=cloud_cover,
            raw_json={},
        )

    with connect(db_path) as conn:
        insert_forecasts(
            conn,
            [
                _record("source_a", other_day, 40.0),
                _record("source_b", other_day, 44.0),
                _record("source_c", other_day, 42.0),
                _record("source_a", target_day, 80.0),
                _record("source_b", target_day, 84.0),
                # source_c has no row at all for target_day - a genuine gap.
            ],
        )
        for d in (other_day, target_day):
            upsert_actual(
                conn,
                ActualRecord(
                    source="open_meteo_archive",
                    location_name=location.name,
                    lat=location.lat,
                    lon=location.lon,
                    actual_date=d,
                    collected_at=datetime.now().replace(microsecond=0),
                    max_temp=20.0,
                    raw_json={},
                ),
            )

    df = build_feature_table(db_path, location)
    row = df[df["forecast_date"] == pd.Timestamp(target_day)].iloc[0]

    assert row["source_c__cloud_cover"] == 82.0  # this row's own median of [80.0, 84.0]
    # The spread stats must reflect only the sources that actually reported,
    # not the filled-in value, or a gap would falsely look like agreement.
    assert row["source_count__cloud_cover"] == 2


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


def test_clip_prediction_floors_non_negative_targets_at_zero():
    assert clip_prediction("precipitation_sum", -0.6) == 0.0
    assert clip_prediction("wind_speed", -1.0) == 0.0
    assert clip_prediction("wind_gusts", -1.0) == 0.0
    assert clip_prediction("precipitation_sum", 2.3) == 2.3


def test_clip_prediction_leaves_temperature_targets_unclipped():
    assert clip_prediction("max_temp", -4.0) == -4.0
    assert clip_prediction("min_temp", -4.0) == -4.0


def test_fill_missing_sources_from_row_median_covers_a_source_absent_from_the_whole_row(tmp_path):
    """A source can be completely absent from a single-day live prediction row
    (no column at all, not just NaN) if it was trained on historically but
    didn't report today - reindex introduces it as all-NaN. This must be
    filled from what the sources that DID report today are saying, not left
    for the pipeline's SimpleImputer to substitute its stale fit-time median.
    """
    # Named after real provider conventions (not "source_a"/"source_b") since
    # the helper deliberately excludes columns literally prefixed "source_" -
    # that's how it tells a genuine per-source column apart from an aggregate
    # stat column like "source_mean__cloud_cover".
    raw = pd.DataFrame({"weatherbit__cloud_cover": [80.0], "wttr_in__cloud_cover": [84.0]})
    # Simulate reindexing to a model trained when bom was still active.
    X = raw.reindex(columns=["weatherbit__cloud_cover", "wttr_in__cloud_cover", "bom__cloud_cover"])
    assert X["bom__cloud_cover"].isna().all()

    filled = _fill_missing_sources_from_row_median(X, raw, "cloud_cover")

    assert filled["bom__cloud_cover"].iloc[0] == 82.0  # median of [80.0, 84.0]
    assert filled["weatherbit__cloud_cover"].iloc[0] == 80.0  # untouched
    assert filled["wttr_in__cloud_cover"].iloc[0] == 84.0  # untouched


def test_clip_prediction_caps_percentage_targets_at_100():
    """A real incident: a missing feature let Ridge extrapolate to 467.9%
    predicted cloud cover for one location on one day - percentages can't
    physically exceed 100."""
    assert clip_prediction("cloud_cover", 467.9) == 100.0
    assert clip_prediction("humidity", 109.3) == 100.0
    assert clip_prediction("cloud_cover", 50.0) == 50.0  # in-range values pass through


def test_clip_prediction_does_not_cap_non_percentage_targets():
    assert clip_prediction("wind_gusts", 150.0) == 150.0
    assert clip_prediction("precipitation_sum", 200.0) == 200.0


def test_predict_latest_ml_clips_negative_precipitation_to_zero(tmp_path):
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
    # A Ridge-like regressor with no non-negativity constraint can legitimately
    # predict a small negative value for a dry day - simulate that directly.
    target = pd.Series([-0.4, -0.4])

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    model.fit(training_frame, target)

    bundle = TrainedModelBundle(
        target="precipitation_sum",
        features=feature_columns,
        model=model,
        metrics={"mae": 1.0},
        trained_at="2026-06-18T00:00:00",
        model_type="regression",
    )
    with (model_dir / "precipitation_sum.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)

    result = predict_latest_ml(db_path, location, model_dir)

    assert result["predictions"]["precipitation_sum"] == 0.0
