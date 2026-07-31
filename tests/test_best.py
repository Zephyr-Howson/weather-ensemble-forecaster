from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from weather_ensemble.backtest import backtest_best_predictions
from weather_ensemble.best import predict_best, select_best_model
from weather_ensemble.config import PERIODS, Location, get_periods_db_path
from weather_ensemble.db import (
    connect,
    connect_periods,
    insert_forecast_periods,
    insert_forecasts,
    upsert_actual,
)
from weather_ensemble.models import ActualRecord, ForecastPeriodRecord, ForecastRecord

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")

SOURCE_ACCURATE = "open_meteo_best_match"
SOURCE_NOISY = "open_meteo_gfs_global"


def _forecast(source: str, forecast_date: date, max_temp: float) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
        max_temp=max_temp,
        min_temp=max_temp - 5,
        precipitation_sum=0.0,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def _actual(actual_date: date, max_temp: float) -> ActualRecord:
    return ActualRecord(
        source="open_meteo_archive",
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        actual_date=actual_date,
        collected_at=datetime.combine(actual_date, datetime.min.time()),
        max_temp=max_temp,
        min_temp=max_temp - 5,
        precipitation_sum=0.0,
        did_rain=0,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def test_select_best_model_requires_min_days():
    """A candidate with a better MAE but too few scored days must lose to a
    worse-but-eligible candidate - a lucky handful of days shouldn't be
    enough to win "Best" for a target.
    """
    dates = pd.date_range("2026-06-01", periods=20, freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"model": "reliable", "target": "max_temp", "forecast_date": d, "abs_error": 1.0})
        if i >= 15:  # only 5 scored days - below the min_days=14 threshold
            rows.append({"model": "lucky_few", "target": "max_temp", "forecast_date": d, "abs_error": 0.1})
    pool = pd.DataFrame(rows)

    model, info = select_best_model(pool, "max_temp", date(2026, 6, 21), window_days=30, min_days=14)
    assert model == "reliable"
    assert info["n"] == 20


def test_select_best_model_picks_lowest_mae_among_eligible():
    dates = pd.date_range("2026-06-01", periods=20, freq="D")
    rows = []
    for d in dates:
        rows.append({"model": "source_a", "target": "max_temp", "forecast_date": d, "abs_error": 0.5})
        rows.append({"model": "source_b", "target": "max_temp", "forecast_date": d, "abs_error": 3.0})
    pool = pd.DataFrame(rows)

    model, info = select_best_model(pool, "max_temp", date(2026, 6, 21), window_days=30, min_days=14)
    assert model == "source_a"
    assert info["mae"] == 0.5


def test_select_best_model_excludes_future_data():
    """The window is strictly before target_date - a row on or after it must
    never influence which candidate wins (no lookahead leakage). A second
    candidate that's perfect but only ever scored on/after target_date must
    lose to a merely-decent candidate with real history before it.
    """
    target_date = date(2026, 6, 11)
    before = pd.date_range("2026-06-01", periods=10, freq="D")  # strictly before target_date
    on_or_after = pd.date_range(target_date, periods=20, freq="D")

    rows = [{"model": "only_real_history", "target": "max_temp", "forecast_date": d, "abs_error": 1.0} for d in before]
    rows += [{"model": "future_perfect", "target": "max_temp", "forecast_date": d, "abs_error": 0.0} for d in on_or_after]
    pool = pd.DataFrame(rows)

    model, info = select_best_model(pool, "max_temp", target_date, window_days=30, min_days=5)
    assert model == "only_real_history"
    assert info["n"] == 10


def test_predict_best_copies_the_winning_candidates_own_value(tmp_path):
    """End-to-end: seed 20 days where one raw source is consistently far more
    accurate than another, then generate tomorrow's Best prediction - it
    should copy the accurate source's own forecast for tomorrow, not the
    noisy one's, even though both are present.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    history_days = [start + timedelta(days=i) for i in range(20)]
    tomorrow = history_days[-1] + timedelta(days=1)

    with connect(db_path) as conn:
        for day in history_days:
            insert_forecasts(
                conn,
                [
                    _forecast(SOURCE_ACCURATE, day, 20.1),  # abs_error ~0.1 vs actual 20.0
                    _forecast(SOURCE_NOISY, day, 25.0),  # abs_error ~5.0 vs actual 20.0
                ],
            )
            upsert_actual(conn, _actual(day, 20.0))
        # Tomorrow: forecasts exist (as a live collection would produce) but no actual yet.
        insert_forecasts(
            conn,
            [
                _forecast(SOURCE_ACCURATE, tomorrow, 21.0),
                _forecast(SOURCE_NOISY, tomorrow, 30.0),
            ],
        )

    result = predict_best(db_path, LOCATION, window_days=30, min_days=14, target_date=tomorrow)

    assert result["chosen_sources"]["max_temp"] == SOURCE_ACCURATE
    assert result["predictions"]["max_temp"] == 21.0

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT forecast_date, max_temp FROM best_predictions WHERE location_name = ?", (LOCATION.name,)
        ).fetchone()
    assert row is not None
    assert row["forecast_date"] == tomorrow.isoformat()
    assert row["max_temp"] == 21.0


def test_backtest_best_predictions_walk_forward(tmp_path):
    """Mirrors test_backtest_predictions_writes_ensemble_and_ml_rows: seed
    history plus one more day with only forecasts (no actual, no existing
    best row) and confirm the walk-forward backcast writes a row for it,
    again drawing from the historically-accurate source.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    days = [start + timedelta(days=i) for i in range(21)]  # 20 days history + 1 target day
    target_day = days[-1]

    with connect(db_path) as conn:
        for day in days:
            insert_forecasts(
                conn,
                [
                    _forecast(SOURCE_ACCURATE, day, 20.1),
                    _forecast(SOURCE_NOISY, day, 25.0),
                ],
            )
            upsert_actual(conn, _actual(day, 20.0))

    result = backtest_best_predictions(db_path, LOCATION, days=1, window_days=30, min_days=14)

    assert result["best"].get("written") == 1

    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT forecast_date, max_temp FROM best_predictions WHERE location_name = ?", (LOCATION.name,)
        ).fetchone()
    assert row is not None
    assert row["forecast_date"] == target_day.isoformat()
    assert row["max_temp"] == 20.1


def _forecast_with_precip(source: str, forecast_date: date, precip: float) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
        max_temp=20.0,
        min_temp=15.0,
        precipitation_sum=precip,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def _actual_with_precip(actual_date: date, precip: float) -> ActualRecord:
    return ActualRecord(
        source="open_meteo_archive",
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        actual_date=actual_date,
        collected_at=datetime.combine(actual_date, datetime.min.time()),
        max_temp=20.0,
        min_temp=15.0,
        precipitation_sum=precip,
        did_rain=int(precip >= 0.2),
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def _forecast_period(source: str, forecast_date: date, period: str, precip: float) -> ForecastPeriodRecord:
    return ForecastPeriodRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        period=period,
        collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
        precipitation_sum=precip,
        rain_probability=50.0,
    )


def test_predict_best_period_precipitation_inherits_daily_winner(tmp_path):
    """Best's 4 period values must come from whichever source won the DAILY
    precipitation_sum selection, copied as-is - never an independent
    per-period pick, or the periods could disagree with Best's own daily
    total (each period potentially crediting a different "winning" source).
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    history_days = [start + timedelta(days=i) for i in range(20)]
    tomorrow = history_days[-1] + timedelta(days=1)

    with connect(db_path) as conn:
        for day in history_days:
            insert_forecasts(
                conn,
                [
                    _forecast_with_precip(SOURCE_ACCURATE, day, 2.1),  # abs_error 0.1 vs actual 2.0
                    _forecast_with_precip(SOURCE_NOISY, day, 8.0),  # abs_error 6.0
                ],
            )
            upsert_actual(conn, _actual_with_precip(day, 2.0))
        insert_forecasts(
            conn,
            [
                _forecast_with_precip(SOURCE_ACCURATE, tomorrow, 3.0),
                _forecast_with_precip(SOURCE_NOISY, tomorrow, 9.0),
            ],
        )

    accurate_periods = dict(zip(PERIODS, [0.5, 1.0, 1.0, 0.5], strict=True))  # sums to 3.0
    noisy_periods = dict(zip(PERIODS, [2.0, 3.0, 2.0, 2.0], strict=True))  # sums to 9.0
    with connect_periods(get_periods_db_path(db_path)) as pconn:
        for period, value in accurate_periods.items():
            insert_forecast_periods(pconn, [_forecast_period(SOURCE_ACCURATE, tomorrow, period, value)])
        for period, value in noisy_periods.items():
            insert_forecast_periods(pconn, [_forecast_period(SOURCE_NOISY, tomorrow, period, value)])

    result = predict_best(db_path, LOCATION, window_days=30, min_days=14, target_date=tomorrow)

    assert result["chosen_sources"]["precipitation_sum"] == SOURCE_ACCURATE
    assert result["predictions"]["precipitation_sum"] == 3.0

    with connect_periods(get_periods_db_path(db_path)) as pconn:
        rows = pconn.execute(
            "SELECT period, precipitation_sum FROM best_predictions_periods WHERE location_name = ? AND forecast_date = ?",
            (LOCATION.name, tomorrow.isoformat()),
        ).fetchall()
    stored = {r["period"]: r["precipitation_sum"] for r in rows}

    assert stored == accurate_periods  # the accurate source's own periods, not the noisy one's
    assert sum(stored.values()) == result["predictions"]["precipitation_sum"]  # periods sum to Best's daily total
