from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import pandas as pd

from weather_ensemble.config import Location
from weather_ensemble.db import connect, insert_forecasts, upsert_actual
from weather_ensemble.models import ActualRecord, ForecastRecord
from weather_ensemble.report import build_html_report
from weather_ensemble.scoring import (
    BASELINE_CLIMATOLOGY,
    BASELINE_PERSISTENCE,
    MODEL_ENSEMBLE,
    MODEL_ML,
    build_predictions_long,
    leaderboard,
    rolling_error_over_time,
)

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")


def _forecast(source: str, forecast_date: date, max_temp: float, precipitation_sum: float = 0.0) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
        max_temp=max_temp,
        min_temp=max_temp - 5,
        precipitation_sum=precipitation_sum,
        uv_index=3.0,
        wind_speed=20.0,
        wind_gusts=30.0,
        raw_json={},
    )


def _actual(actual_date: date, max_temp: float, precipitation_sum: float = 0.0) -> ActualRecord:
    return ActualRecord(
        source="open_meteo_archive",
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        actual_date=actual_date,
        collected_at=datetime.combine(actual_date, datetime.min.time()),
        max_temp=max_temp,
        min_temp=max_temp - 5,
        precipitation_sum=precipitation_sum,
        did_rain=int(precipitation_sum >= 0.2),
        uv_index=3.0,
        wind_speed=20.0,
        wind_gusts=30.0,
    )


def _seed_db(db_path, num_days: int = 10) -> date:
    start = date(2026, 1, 1)
    with connect(db_path) as conn:
        for i in range(num_days):
            day = start + timedelta(days=i)
            upsert_actual(conn, _actual(day, max_temp=20.0 + i, precipitation_sum=0.0 if i % 3 else 5.0))
            insert_forecasts(
                conn,
                [
                    _forecast("open_meteo_best_match", day, max_temp=21.0 + i, precipitation_sum=0.0 if i % 3 else 4.0),
                    _forecast("open_meteo_gfs_global", day, max_temp=19.0 + i, precipitation_sum=0.0 if i % 3 else 6.0),
                ],
            )
            conn.execute(
                """
                INSERT INTO ensemble_predictions (
                    location_name, lat, lon, forecast_date, generated_at, window_days,
                    max_temp, min_temp, precipitation_sum, did_rain, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    LOCATION.name, LOCATION.lat, LOCATION.lon, day.isoformat(),
                    datetime.combine(day, datetime.min.time()).isoformat(), 30,
                    20.0 + i, 15.0 + i, 0.0 if i % 3 else 5.0, 0 if i % 3 else 1, json.dumps({}),
                ),
            )
            conn.execute(
                """
                INSERT INTO ml_predictions (
                    location_name, lat, lon, forecast_date, generated_at, model_version,
                    max_temp, min_temp, precipitation_sum, did_rain, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    LOCATION.name, LOCATION.lat, LOCATION.lon, day.isoformat(),
                    datetime.combine(day, datetime.min.time()).isoformat(), "test-v1",
                    20.5 + i, 15.5 + i, 0.0 if i % 3 else 5.0, 0 if i % 3 else 1, json.dumps({}),
                ),
            )
    return start


def test_build_predictions_long_includes_every_model(tmp_path):
    db_path = tmp_path / "weather.db"
    _seed_db(db_path)

    long_df = build_predictions_long(db_path, [LOCATION])

    models = set(long_df["model"].unique())
    assert {"open_meteo_best_match", "open_meteo_gfs_global", MODEL_ENSEMBLE, MODEL_ML, BASELINE_PERSISTENCE, BASELINE_CLIMATOLOGY} <= models
    assert "did_rain" in set(long_df["target"].unique())
    # Raw sources don't store did_rain directly - it must be derived from precipitation_sum.
    raw_rain_rows = long_df[(long_df["model"] == "open_meteo_best_match") & (long_df["target"] == "did_rain")]
    assert not raw_rain_rows.empty
    assert set(raw_rain_rows["predicted"].unique()) <= {0.0, 1.0}


def test_persistence_and_climatology_use_only_past_data(tmp_path):
    db_path = tmp_path / "weather.db"
    start = _seed_db(db_path, num_days=10)

    long_df = build_predictions_long(db_path, [LOCATION])
    persistence = long_df[(long_df["model"] == BASELINE_PERSISTENCE) & (long_df["target"] == "max_temp")]
    persistence = persistence.sort_values("forecast_date")

    # Day 0 has no prior actual, so persistence can't predict it.
    assert (start + timedelta(days=0)) not in {d.date() for d in persistence["forecast_date"]}
    # Day i's persistence prediction should equal actual max_temp on day i-1 (20 + (i-1)).
    row = persistence[persistence["forecast_date"] == pd.Timestamp(start + timedelta(days=5))]
    assert row["predicted"].iloc[0] == 20.0 + 4


def test_leaderboard_ranks_lower_mae_first(tmp_path):
    db_path = tmp_path / "weather.db"
    _seed_db(db_path)
    long_df = build_predictions_long(db_path, [LOCATION])

    board = leaderboard(long_df, recent_days=None)
    max_temp_board = board[board["target"] == "max_temp"].reset_index(drop=True)
    assert list(max_temp_board["mae"]) == sorted(max_temp_board["mae"])


def test_rolling_error_over_time_produces_one_row_per_date(tmp_path):
    db_path = tmp_path / "weather.db"
    _seed_db(db_path)
    long_df = build_predictions_long(db_path, [LOCATION])

    trend = rolling_error_over_time(long_df, window=3)
    ensemble_max_temp = trend[(trend["model"] == MODEL_ENSEMBLE) & (trend["target"] == "max_temp")]
    assert ensemble_max_temp["forecast_date"].is_unique
    assert (ensemble_max_temp["rolling_mae"] >= 0).all()


def test_build_html_report_writes_file(tmp_path):
    db_path = tmp_path / "weather.db"
    _seed_db(db_path)
    long_df = build_predictions_long(db_path, [LOCATION])

    output = build_html_report(long_df, tmp_path / "report.html", db_path)

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("<html>") or "<div" in output.read_text(encoding="utf-8")


def test_build_html_report_handles_empty_input(tmp_path):
    output = build_html_report(pd.DataFrame(), tmp_path / "empty.html", tmp_path / "empty.db")
    assert output.exists()
