from __future__ import annotations

from datetime import date, datetime, timedelta

from weather_ensemble.backtest import backtest_predictions
from weather_ensemble.config import Location
from weather_ensemble.db import connect, insert_forecasts, upsert_actual
from weather_ensemble.models import ActualRecord, ForecastRecord

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")


def _forecast(source: str, forecast_date: date, value: float) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
        max_temp=value,
        min_temp=value - 5,
        precipitation_sum=0.0,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def _actual(actual_date: date, value: float) -> ActualRecord:
    return ActualRecord(
        source="open_meteo_archive",
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        actual_date=actual_date,
        collected_at=datetime.combine(actual_date, datetime.min.time()),
        max_temp=value,
        min_temp=value - 5,
        precipitation_sum=0.0,
        did_rain=0,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def test_backtest_predictions_writes_ensemble_and_ml_rows(tmp_path):
    """End-to-end: seed enough forecast+actual history (backtest.py's ML step
    needs >= MIN_TRAIN_ROWS=30 rows of prior history to attempt training) and
    run a real backtest for the most recent day. This exists because
    backtest.py's INSERT statements had zero test coverage and a real
    column/placeholder count mismatch (17 columns, 16 placeholders) went
    undetected until it broke every location's regeneration at once -
    asserting on the actual written rows, not just "no exception", would have
    caught it immediately.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 5, 1)
    days = [start + timedelta(days=i) for i in range(32)]  # 31 days history + 1 target day
    target_day = days[-1]

    with connect(db_path) as conn:
        for i, day in enumerate(days):
            value = 15.0 + i * 0.1
            insert_forecasts(
                conn,
                [
                    _forecast("open_meteo_best_match", day, value),
                    _forecast("open_meteo_gfs_global", day, value + 1.0),
                ],
            )
            upsert_actual(conn, _actual(day, value + 0.5))

    result = backtest_predictions(db_path, LOCATION, days=1, ensemble_window_days=30, train_window_days=90)

    assert result["ensemble"].get("written") == 1
    assert result["ml"].get("written") == 1

    with connect(db_path) as conn:
        ensemble_row = conn.execute(
            "SELECT forecast_date, max_temp, wind_speed, pressure_msl FROM ensemble_predictions WHERE location_name = ?",
            (LOCATION.name,),
        ).fetchone()
        ml_row = conn.execute(
            "SELECT forecast_date, max_temp, wind_speed, pressure_msl FROM ml_predictions WHERE location_name = ?",
            (LOCATION.name,),
        ).fetchone()

    assert ensemble_row is not None
    assert ensemble_row["forecast_date"] == target_day.isoformat()
    assert ensemble_row["max_temp"] is not None

    assert ml_row is not None
    assert ml_row["forecast_date"] == target_day.isoformat()
