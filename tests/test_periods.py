from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from weather_ensemble.backtest import backtest_period_predictions
from weather_ensemble.config import Location, get_periods_db_path
from weather_ensemble.db import connect_periods, insert_forecast_periods, upsert_actual_period
from weather_ensemble.ml import predict_latest_ml_period, train_period_model
from weather_ensemble.models import ActualPeriodRecord, ForecastPeriodRecord
from weather_ensemble.service import blend_forecast_period, compute_period_mae_scores

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")


def _forecast_period(source: str, forecast_date: date, period: str, precip: float) -> ForecastPeriodRecord:
    return ForecastPeriodRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        period=period,
        collected_at=datetime.now(UTC).replace(microsecond=0, tzinfo=None),
        precipitation_sum=precip,
        rain_probability=50.0,
    )


def _actual_period(actual_date: date, period: str, precip: float) -> ActualPeriodRecord:
    return ActualPeriodRecord(
        source="open_meteo_archive",
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        actual_date=actual_date,
        period=period,
        collected_at=datetime.now(UTC).replace(microsecond=0, tzinfo=None),
        precipitation_sum=precip,
        did_rain=int(precip >= 0.2),
    )


def test_insert_forecast_periods_and_upsert_actual_period_roundtrip(tmp_path):
    db_path = tmp_path / "weather.db"
    day = date(2026, 7, 28)

    with connect_periods(get_periods_db_path(db_path)) as conn:
        n = insert_forecast_periods(conn, [_forecast_period("open_meteo_best_match", day, "morning", 1.2)])
        assert n == 1
        upsert_actual_period(conn, _actual_period(day, "morning", 0.8))

        row = conn.execute("SELECT * FROM forecast_periods").fetchone()
        assert row["period"] == "morning"
        assert row["precipitation_sum"] == 1.2

        actual_row = conn.execute("SELECT * FROM actual_periods").fetchone()
        assert actual_row["precipitation_sum"] == 0.8
        assert actual_row["did_rain"] == 1


def test_upsert_actual_period_updates_existing_row_not_a_duplicate(tmp_path):
    db_path = tmp_path / "weather.db"
    day = date(2026, 7, 28)

    with connect_periods(get_periods_db_path(db_path)) as conn:
        upsert_actual_period(conn, _actual_period(day, "evening", 0.0))
        upsert_actual_period(conn, _actual_period(day, "evening", 4.5))  # corrected/re-collected value

        rows = conn.execute("SELECT * FROM actual_periods").fetchall()
        assert len(rows) == 1
        assert rows[0]["precipitation_sum"] == 4.5
        assert rows[0]["did_rain"] == 1


def test_compute_period_mae_scores_uses_true_observed_targets():
    df = pd.DataFrame(
        {
            "source": ["a", "a", "b"],
            "precipitation_sum": [0.0, 2.0, 5.0],
            "actual_precipitation_sum": [1.0, 1.0, 3.0],
        }
    )
    scores = compute_period_mae_scores(df)
    assert scores["a"] == 1.0
    assert scores["b"] == 2.0


def test_blend_forecast_period_reconstruction_excludes_lookahead_history(tmp_path):
    """Mirrors test_blend_forecast.py's identical daily-level test: MAE
    weighting for a reconstructed target_date must only use period history
    strictly before it, or a source's *later* accuracy would leak into a
    weight it couldn't have earned yet at that point in time.
    """
    db_path = tmp_path / "weather.db"
    period = "afternoon"
    target = date(2026, 6, 10)
    before1, before2, after = date(2026, 6, 8), date(2026, 6, 9), date(2026, 6, 11)
    actual_value = 5.0

    with connect_periods(get_periods_db_path(db_path)) as conn:
        insert_forecast_periods(
            conn,
            [
                _forecast_period("good_source", before1, period, 5.1),  # error 0.1
                _forecast_period("good_source", before2, period, 5.1),  # error 0.1
                _forecast_period("good_source", after, period, 0.0),  # error 5 (would only matter if leaked)
                _forecast_period("good_source", target, period, 5.2),  # the prediction being blended
                _forecast_period("late_bloomer", before1, period, 0.0),  # error 5
                _forecast_period("late_bloomer", before2, period, 0.0),  # error 5
                _forecast_period("late_bloomer", after, period, 5.0),  # error 0 (would only matter if leaked)
                _forecast_period("late_bloomer", target, period, 0.0),  # the prediction being blended
            ],
        )
        for d in (before1, before2, after):
            upsert_actual_period(conn, _actual_period(d, period, actual_value))

    result = blend_forecast_period(db_path, LOCATION, period, window_days=3650, target_date=target)

    # If the post-target_date day leaked into the weighting, the blend would
    # fall closer to late_bloomer's forecast (0.0), well below this threshold.
    assert result["precipitation_sum"] > 4.0


def test_train_and_predict_period_model_end_to_end(tmp_path):
    """Seeds enough period history to clear MIN_TRAIN_ROWS-equivalent, trains
    a real Ridge model for one period, and predicts tomorrow - asserting on
    the actual written ml_predictions_periods row, not just "no exception"
    (the exact class of bug that went undetected in backtest.py's INSERT
    before it had any test coverage).
    """
    db_path = tmp_path / "weather.db"
    model_dir = tmp_path / "models"
    period = "morning"
    start = date(2026, 5, 1)
    days = [start + timedelta(days=i) for i in range(35)]  # comfortably over MIN_TRAIN_ROWS=30
    tomorrow = days[-1] + timedelta(days=1)

    with connect_periods(get_periods_db_path(db_path)) as conn:
        for i, day in enumerate(days):
            precip = 0.0 if i % 4 else 2.0 + i * 0.05
            insert_forecast_periods(
                conn,
                [
                    _forecast_period("open_meteo_ecmwf_ifs025", day, period, precip + 0.1),
                    _forecast_period("open_meteo_gfs_global", day, period, precip - 0.1),
                ],
            )
            upsert_actual_period(conn, _actual_period(day, period, precip))
        # Tomorrow's live forecast, with no actual yet.
        insert_forecast_periods(
            conn,
            [
                _forecast_period("open_meteo_ecmwf_ifs025", tomorrow, period, 1.5),
                _forecast_period("open_meteo_gfs_global", tomorrow, period, 1.3),
            ],
        )

    train_result = train_period_model(db_path, LOCATION, period, model_dir, min_rows=30)
    assert train_result["status"] == "trained"
    assert train_result["train_rows"] + train_result["test_rows"] == len(days)

    predict_result = predict_latest_ml_period(db_path, LOCATION, period, model_dir, target_date=tomorrow)
    assert predict_result["forecast_date"] == tomorrow.isoformat()
    assert predict_result["precipitation_sum"] >= 0.0  # clip_prediction floors negative Ridge output at 0

    with connect_periods(get_periods_db_path(db_path)) as conn:
        row = conn.execute(
            "SELECT forecast_date, period, precipitation_sum FROM ml_predictions_periods WHERE location_name = ?",
            (LOCATION.name,),
        ).fetchone()
    assert row is not None
    assert row["forecast_date"] == tomorrow.isoformat()
    assert row["period"] == period
    assert row["precipitation_sum"] == predict_result["precipitation_sum"]


def test_backtest_period_predictions_writes_ensemble_and_ml_rows(tmp_path):
    """Mirrors test_backtest.py's identical daily-level test. This one matters
    for a distinct real reason: a live "tomorrow" prediction has no actual
    yet to score against, so without walk-forward historical predictions
    like these, the report's period leaderboard cards can only ever show raw
    source accuracy - the ensemble/ML entries were confirmed missing from
    the rendered report before backtest_period_predictions was added.
    """
    db_path = tmp_path / "weather.db"
    period = "afternoon"
    start = date(2026, 5, 1)
    days = [start + timedelta(days=i) for i in range(32)]  # 31 days history + 1 target day
    target_day = days[-1]

    with connect_periods(get_periods_db_path(db_path)) as conn:
        for i, day in enumerate(days):
            precip = 0.0 if i % 3 else 2.0 + i * 0.05
            insert_forecast_periods(
                conn,
                [
                    _forecast_period("open_meteo_best_match", day, period, precip + 0.1),
                    _forecast_period("open_meteo_gfs_global", day, period, precip - 0.1),
                ],
            )
            upsert_actual_period(conn, _actual_period(day, period, precip))

    result = backtest_period_predictions(db_path, LOCATION, period, days=1, ensemble_window_days=30, train_window_days=90)

    assert result["ensemble"].get("written") == 1
    assert result["ml"].get("written") == 1

    with connect_periods(get_periods_db_path(db_path)) as conn:
        ensemble_row = conn.execute(
            "SELECT forecast_date, period, precipitation_sum FROM ensemble_predictions_periods WHERE location_name = ?",
            (LOCATION.name,),
        ).fetchone()
        ml_row = conn.execute(
            "SELECT forecast_date, period, precipitation_sum FROM ml_predictions_periods WHERE location_name = ?",
            (LOCATION.name,),
        ).fetchone()

    assert ensemble_row is not None
    assert ensemble_row["forecast_date"] == target_day.isoformat()
    assert ensemble_row["period"] == period
    assert ensemble_row["precipitation_sum"] is not None

    assert ml_row is not None
    assert ml_row["forecast_date"] == target_day.isoformat()
    assert ml_row["period"] == period
