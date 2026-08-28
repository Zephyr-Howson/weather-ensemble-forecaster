from __future__ import annotations

from datetime import date, timedelta
from functools import partial

from weather_ensemble import cli
from weather_ensemble.config import Location, local_today
from weather_ensemble.db import connect
from weather_ensemble.service import default_forecast_target_date, missing_ensemble_dates

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")


def _insert_ensemble_row(db_path, forecast_date: date) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO ensemble_predictions (location_name, lat, lon, forecast_date, generated_at, window_days, max_temp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (LOCATION.name, LOCATION.lat, LOCATION.lon, forecast_date.isoformat(), "2026-01-01T09:00:00", 30, 18.0),
        )
        conn.commit()


def test_default_forecast_target_date_with_no_history_returns_wall_clock_tomorrow(tmp_path):
    db_path = tmp_path / "weather.db"
    connect(db_path).close()

    result = default_forecast_target_date(db_path, LOCATION)

    assert result == local_today(LOCATION) + timedelta(days=1)


def test_default_forecast_target_date_matches_wall_clock_tomorrow_when_on_schedule(tmp_path):
    """Steady state: last night's run already produced today's ensemble row,
    so continuity (last + 1) and wall-clock (today + 1) agree."""
    db_path = tmp_path / "weather.db"
    connect(db_path).close()
    _insert_ensemble_row(db_path, local_today(LOCATION))

    result = default_forecast_target_date(db_path, LOCATION)

    assert result == local_today(LOCATION) + timedelta(days=1)


def test_default_forecast_target_date_caps_at_the_gap_instead_of_skipping_ahead(tmp_path):
    """Reproduces the 2026-08-27 incident: the last real ensemble prediction
    is "yesterday" (the previous run never advanced past it, e.g. because a
    delayed run's collection landed on the wrong wall-clock date), so the
    correct next date is "today" - not wall-clock "tomorrow", which would
    silently skip today's date the way the real incident skipped 2026-08-28.
    """
    db_path = tmp_path / "weather.db"
    connect(db_path).close()
    _insert_ensemble_row(db_path, local_today(LOCATION) - timedelta(days=1))

    result = default_forecast_target_date(db_path, LOCATION)

    assert result == local_today(LOCATION)
    assert result != local_today(LOCATION) + timedelta(days=1)


def test_missing_ensemble_dates_empty_for_a_brand_new_location(tmp_path):
    db_path = tmp_path / "weather.db"
    connect(db_path).close()

    assert missing_ensemble_dates(db_path, LOCATION) == []


def test_missing_ensemble_dates_finds_a_gap_hidden_behind_a_later_date(tmp_path):
    """A MAX(forecast_date)-only check would miss this: today-2 is missing,
    but today-1 already exists (e.g. from a delayed run that jumped past it),
    so the gap is only visible by scanning the whole recent window."""
    db_path = tmp_path / "weather.db"
    connect(db_path).close()
    today = local_today(LOCATION)
    _insert_ensemble_row(db_path, today - timedelta(days=3))
    _insert_ensemble_row(db_path, today - timedelta(days=1))

    missing = missing_ensemble_dates(db_path, LOCATION)

    assert today - timedelta(days=2) in missing
    assert today - timedelta(days=3) not in missing
    assert today - timedelta(days=1) not in missing
    assert today not in missing  # today itself is never "missing" - it hasn't happened yet


def test_missing_ensemble_dates_bounded_by_lookback_window(tmp_path):
    """A location down for longer than CATCH_UP_LOOKBACK_DAYS shouldn't grow
    an ever-larger auto-reconstruction; only recent gaps are reported."""
    db_path = tmp_path / "weather.db"
    connect(db_path).close()
    today = local_today(LOCATION)
    _insert_ensemble_row(db_path, today - timedelta(days=60))

    missing = missing_ensemble_dates(db_path, LOCATION)

    assert today - timedelta(days=60) not in missing
    assert len(missing) == 10  # CATCH_UP_LOOKBACK_DAYS


def test_catch_up_missed_forecasts_is_a_noop_with_no_gap(tmp_path, monkeypatch):
    db_path = tmp_path / "weather.db"
    connect(db_path).close()

    def _unexpected_call(*_a, _name, **_k):
        raise AssertionError(f"{_name} should not be called")

    for name in ("backfill", "backtest_predictions", "backtest_best_predictions", "generate_and_store_best_narrative"):
        monkeypatch.setattr(cli, name, partial(_unexpected_call, _name=name))

    parser = cli.build_parser()
    args = parser.parse_args(["--db", str(db_path), "--collect"])

    result = cli._catch_up_missed_forecasts(args, LOCATION)

    assert result == {"missing_dates": []}


def test_catch_up_missed_forecasts_backfills_and_reconstructs_the_gap(tmp_path, monkeypatch):
    db_path = tmp_path / "weather.db"
    connect(db_path).close()
    today = local_today(LOCATION)
    for i in range(1, 11):  # fill the whole lookback window except today-2
        if i != 2:
            _insert_ensemble_row(db_path, today - timedelta(days=i))

    calls = {}
    monkeypatch.setattr(cli, "backfill", lambda db, loc, days: calls.setdefault("backfill", days))
    monkeypatch.setattr(
        cli, "backtest_predictions", lambda db, loc, days, ensemble_window_days, train_window_days: calls.setdefault("backtest", days)
    )
    monkeypatch.setattr(
        cli, "backtest_best_predictions", lambda db, loc, days, window_days, min_days: calls.setdefault("backtest_best", days)
    )
    monkeypatch.setattr(
        cli, "generate_and_store_best_narrative", lambda db, loc, target_date: {"narrative": f"narrative for {target_date}"}
    )

    parser = cli.build_parser()
    args = parser.parse_args(["--db", str(db_path), "--collect"])

    result = cli._catch_up_missed_forecasts(args, LOCATION)

    assert result["missing_dates"] == [(today - timedelta(days=2)).isoformat()]
    assert result["gap_days"] == 2  # (today - (today-2)).days
    assert calls["backfill"] == 2
    assert calls["backtest"] == 2
    assert calls["backtest_best"] == 2
    assert result["narrated"] == [(today - timedelta(days=2)).isoformat()]
