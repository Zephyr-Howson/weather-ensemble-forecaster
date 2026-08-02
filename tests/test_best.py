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


def test_predict_best_falls_through_when_top_candidate_has_no_value_for_the_date(tmp_path):
    """Even a candidate that's been the most accurate all month can have a
    one-off collection gap on the exact date being predicted - Best must
    fall through to the next-best eligible candidate rather than leaving
    the target unfilled just because its #1 pick was silent today.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    history_days = [start + timedelta(days=i) for i in range(20)]
    tomorrow = history_days[-1] + timedelta(days=1)

    top_performer = "open_meteo_ecmwf_ifs025"
    second_best = "open_meteo_gem_seamless"

    with connect(db_path) as conn:
        for day in history_days:
            insert_forecasts(
                conn,
                [
                    _forecast(top_performer, day, 20.05),  # abs_error ~0.05 - clearly the best all month
                    _forecast(second_best, day, 20.5),  # abs_error ~0.5 - clearly second
                ],
            )
            upsert_actual(conn, _actual(day, 20.0))
        # Tomorrow: the top performer has a one-off collection gap (no forecast
        # row at all) - only the second-best source actually has a value.
        insert_forecasts(conn, [_forecast(second_best, tomorrow, 21.0)])

    result = predict_best(db_path, LOCATION, window_days=30, min_days=14, target_date=tomorrow)

    assert result["chosen_sources"]["max_temp"] == second_best
    assert result["predictions"]["max_temp"] == 21.0


def _forecast_precip_and_rain(
    source: str, forecast_date: date, precip: float, rain_probability: float
) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
        max_temp=20.0,
        min_temp=15.0,
        rain_probability=rain_probability,
        precipitation_sum=precip,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def test_predict_best_precipitation_excludes_non_period_aware_candidates(tmp_path):
    """A non-period-aware source (BOM here - see PERIOD_AWARE_CANDIDATES in
    best.py) must never win Best's precipitation_sum selection even with a
    far better MAE than every period-aware candidate, since Best's period
    breakdown can only ever be copied from a period-aware winner (see
    _period_values_for_model - no other provider ever collects sub-daily rain
    data at all). The same source is still free to win an unrelated target
    (max_temp here) on its own merits - the restriction only scopes
    precipitation_sum/did_rain, not the source everywhere.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    history_days = [start + timedelta(days=i) for i in range(20)]
    tomorrow = history_days[-1] + timedelta(days=1)

    def forecast(source: str, forecast_date: date, max_temp: float, precip: float) -> ForecastRecord:
        return ForecastRecord(
            source=source,
            location_name=LOCATION.name,
            lat=LOCATION.lat,
            lon=LOCATION.lon,
            forecast_date=forecast_date,
            collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
            max_temp=max_temp,
            min_temp=max_temp - 5,
            precipitation_sum=precip,
            wind_speed=15.0,
            wind_gusts=25.0,
            cloud_cover=40.0,
            humidity=60.0,
            pressure_msl=1015.0,
            raw_json={},
        )

    non_period_source = "bom"

    with connect(db_path) as conn:
        for day in history_days:
            insert_forecasts(
                conn,
                [
                    # bom: near-perfect precipitation, but a poor max_temp forecast.
                    forecast(non_period_source, day, 20.1, 2.01),
                    # the only period-aware candidate: worse precipitation MAE,
                    # but a near-perfect max_temp forecast.
                    forecast(SOURCE_NOISY, day, 30.0, 4.0),
                ],
            )
            upsert_actual(conn, _actual_with_precip(day, 2.0))  # actual max_temp=20.0, precip=2.0
        insert_forecasts(
            conn,
            [
                forecast(non_period_source, tomorrow, 21.0, 5.0),
                forecast(SOURCE_NOISY, tomorrow, 31.0, 6.0),
            ],
        )

    result = predict_best(db_path, LOCATION, window_days=30, min_days=14, target_date=tomorrow)

    # bom has the far better precipitation MAE (~0.01 vs ~2.0) but is excluded
    # from precipitation_sum/did_rain selection entirely.
    assert result["chosen_sources"]["precipitation_sum"] == SOURCE_NOISY
    assert result["predictions"]["precipitation_sum"] == 6.0
    # ...yet bom still wins max_temp on its own merits - the restriction is
    # scoped to precipitation_sum/did_rain only.
    assert result["chosen_sources"]["max_temp"] == non_period_source
    assert result["predictions"]["max_temp"] == 21.0


def test_predict_best_did_rain_copied_from_precipitation_winner(tmp_path):
    """did_rain must come from whichever period-aware candidate wins
    precipitation_sum, never from an independent per-target did_rain
    ranking - even when a different candidate has a far better did_rain MAE
    of its own. Picking rain%, precipitation mm, and every period from the
    same source is the whole point of linking the two selections together.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    history_days = [start + timedelta(days=i) for i in range(20)]
    tomorrow = history_days[-1] + timedelta(days=1)

    precip_winner = "open_meteo_ecmwf_ifs025"  # best precipitation MAE, poor rain% MAE
    rain_winner = "open_meteo_gem_seamless"  # worse precipitation MAE, best rain% MAE

    with connect(db_path) as conn:
        for day in history_days:
            insert_forecasts(
                conn,
                [
                    _forecast_precip_and_rain(precip_winner, day, 2.01, 10.0),  # precip abs_error ~0.01
                    _forecast_precip_and_rain(rain_winner, day, 4.0, 95.0),  # rain% abs_error 0.05 vs actual 1.0
                ],
            )
            upsert_actual(conn, _actual_with_precip(day, 2.0))  # did_rain actual = 1 (precip >= 0.2)
        insert_forecasts(
            conn,
            [
                _forecast_precip_and_rain(precip_winner, tomorrow, 5.0, 15.0),
                _forecast_precip_and_rain(rain_winner, tomorrow, 6.0, 90.0),
            ],
        )

    result = predict_best(db_path, LOCATION, window_days=30, min_days=14, target_date=tomorrow)

    assert result["chosen_sources"]["precipitation_sum"] == precip_winner
    assert result["predictions"]["precipitation_sum"] == 5.0
    # did_rain must be copied from precip_winner (0.15), not rain_winner
    # (0.90), even though rain_winner has the better standalone did_rain MAE.
    assert result["chosen_sources"]["did_rain"] == precip_winner
    assert result["predictions"]["did_rain_probability"] == 0.15


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


def test_predict_best_precipitation_falls_through_when_winner_lacks_periods_for_this_date(tmp_path):
    """A period-aware candidate can still have a one-off gap in its own
    period collection for this exact date - e.g. a brand-new live-only
    source whose period tracking came online a day after its daily forecasts
    did (confirmed in practice for open_meteo_best_match). Best must skip it
    in favor of the next-ranked period-aware candidate that actually has
    period data for this date, even though the skipped candidate has the
    better standalone precipitation MAE - a complete winner beats a more
    "accurate" one that would leave every period blank.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    history_days = [start + timedelta(days=i) for i in range(20)]
    tomorrow = history_days[-1] + timedelta(days=1)

    no_periods_today = "open_meteo_best_match"  # best precip MAE, but no periods seeded for tomorrow
    has_periods_today = "open_meteo_ecmwf_ifs025"  # worse precip MAE, but has periods seeded for tomorrow

    with connect(db_path) as conn:
        for day in history_days:
            insert_forecasts(
                conn,
                [
                    _forecast_with_precip(no_periods_today, day, 2.01),  # abs_error ~0.01
                    _forecast_with_precip(has_periods_today, day, 4.0),  # abs_error 2.0
                ],
            )
            upsert_actual(conn, _actual_with_precip(day, 2.0))
        insert_forecasts(
            conn,
            [
                _forecast_with_precip(no_periods_today, tomorrow, 5.0),
                _forecast_with_precip(has_periods_today, tomorrow, 6.0),
            ],
        )

    # Only has_periods_today gets period rows seeded for tomorrow - simulates
    # no_periods_today's period collection not having started yet for this date.
    complete_periods = dict(zip(PERIODS, [0.5, 1.0, 1.0, 0.5], strict=True))
    with connect_periods(get_periods_db_path(db_path)) as pconn:
        for period, value in complete_periods.items():
            insert_forecast_periods(pconn, [_forecast_period(has_periods_today, tomorrow, period, value)])

    result = predict_best(db_path, LOCATION, window_days=30, min_days=14, target_date=tomorrow)

    assert result["chosen_sources"]["precipitation_sum"] == has_periods_today
    assert result["predictions"]["precipitation_sum"] == 6.0

    with connect_periods(get_periods_db_path(db_path)) as pconn:
        rows = pconn.execute(
            "SELECT period, precipitation_sum FROM best_predictions_periods WHERE location_name = ? AND forecast_date = ?",
            (LOCATION.name, tomorrow.isoformat()),
        ).fetchall()
    stored = {r["period"]: r["precipitation_sum"] for r in rows}
    assert stored == complete_periods


def test_predict_best_precipitation_falls_back_to_value_only_when_no_candidate_has_periods(tmp_path):
    """If literally no period-aware candidate has period data for this exact
    date, Best must still report its best available precipitation_sum/
    did_rain figure rather than losing the day's prediction entirely - the
    same graceful-degradation guarantee every other target already gets.
    Periods are left blank in this genuine last-resort case.
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
    # No period rows seeded for either source at all.

    result = predict_best(db_path, LOCATION, window_days=30, min_days=14, target_date=tomorrow)

    assert result["chosen_sources"]["precipitation_sum"] == SOURCE_ACCURATE
    assert result["predictions"]["precipitation_sum"] == 3.0

    with connect_periods(get_periods_db_path(db_path)) as pconn:
        rows = pconn.execute(
            "SELECT period FROM best_predictions_periods WHERE location_name = ? AND forecast_date = ?",
            (LOCATION.name, tomorrow.isoformat()),
        ).fetchall()
    assert rows == []
