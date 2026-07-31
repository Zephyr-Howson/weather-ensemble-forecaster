from __future__ import annotations

from datetime import date, datetime, timedelta

from weather_ensemble.config import Location
from weather_ensemble.data_quality import compute_data_quality
from weather_ensemble.db import connect, insert_forecasts
from weather_ensemble.models import ForecastRecord

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")


def _forecast(source: str, forecast_date: date, precipitation_sum: float | None) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()),
        max_temp=20.0,
        min_temp=15.0,
        precipitation_sum=precipitation_sum,
        wind_speed=15.0,
        wind_gusts=25.0,
        cloud_cover=40.0,
        humidity=60.0,
        pressure_msl=1015.0,
        raw_json={},
    )


def test_structurally_unavailable_field_is_not_counted_as_unexpected_null(tmp_path):
    """A source that NEVER reports a field (every row null) is exercising a
    known limitation, not an anomaly - it must not inflate the null rate.
    A source that reports the field on most days but is null on exactly one
    real anomaly day should show up clearly.
    """
    db_path = tmp_path / "weather.db"
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(20)]

    with connect(db_path) as conn:
        for day in days:
            insert_forecasts(
                conn,
                [
                    _forecast("never_reports_precip", day, None),  # always null - structural
                    _forecast("one_anomaly", day, 1.0 if day != days[5] else None),  # null on exactly 1 of 20 days
                ],
            )

    result = compute_data_quality(db_path, [LOCATION.name], window_days=30)
    entities = {e["name"]: e for e in result["Melbourne"]["entities"]}

    assert entities["never_reports_precip"]["null_pct"] == 0.0
    assert entities["one_anomaly"]["null_pct"] > 0.0


def test_missing_rows_not_counted_before_source_first_appeared(tmp_path):
    """A source that only started reporting partway through the window must
    not be penalized for the days before it existed - only genuine gaps
    within its own active range count as missing.
    """
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    all_days = [start + timedelta(days=i) for i in range(20)]
    late_starter_days = all_days[10:]  # only reports for the last 10 of 20 days, with zero gaps in that range

    with connect(db_path) as conn:
        for day in all_days:
            insert_forecasts(conn, [_forecast("always_present", day, 1.0)])
        for day in late_starter_days:
            insert_forecasts(conn, [_forecast("late_starter", day, 1.0)])

    result = compute_data_quality(db_path, [LOCATION.name], window_days=30)
    entities = {e["name"]: e for e in result["Melbourne"]["entities"]}

    assert entities["always_present"]["missing_pct"] == 0.0
    assert entities["late_starter"]["missing_pct"] == 0.0  # complete within ITS OWN active range


def test_missing_rows_detects_a_real_gap_within_active_range(tmp_path):
    db_path = tmp_path / "weather.db"
    start = date(2026, 6, 1)
    days = [start + timedelta(days=i) for i in range(20)]
    gap_day = days[10]

    with connect(db_path) as conn:
        for day in days:
            if day == gap_day:
                continue  # a real, isolated gap in an otherwise-continuous source
            insert_forecasts(conn, [_forecast("mostly_reliable", day, 1.0)])

    result = compute_data_quality(db_path, [LOCATION.name], window_days=30)
    entities = {e["name"]: e for e in result["Melbourne"]["entities"]}

    assert entities["mostly_reliable"]["missing_pct"] > 0.0


def test_pooled_all_matches_combined_totals_across_locations(tmp_path):
    db_path = tmp_path / "weather.db"
    other = Location(name="Sydney", lat=-33.8688, lon=151.2093, timezone="Australia/Sydney")
    days = [date(2026, 6, 1) + timedelta(days=i) for i in range(20)]

    with connect(db_path) as conn:
        for day in days:
            insert_forecasts(
                conn,
                [
                    ForecastRecord(
                        source="src",
                        location_name=LOCATION.name,
                        lat=LOCATION.lat,
                        lon=LOCATION.lon,
                        forecast_date=day,
                        collected_at=datetime.combine(day - timedelta(days=1), datetime.min.time()),
                        max_temp=20.0, min_temp=15.0, precipitation_sum=1.0,
                        wind_speed=15.0, wind_gusts=25.0, cloud_cover=40.0, humidity=60.0, pressure_msl=1015.0,
                        raw_json={},
                    ),
                ],
            )
            if day != days[5]:  # Sydney has one real gap day
                insert_forecasts(
                    conn,
                    [
                        ForecastRecord(
                            source="src",
                            location_name=other.name,
                            lat=other.lat,
                            lon=other.lon,
                            forecast_date=day,
                            collected_at=datetime.combine(day - timedelta(days=1), datetime.min.time()),
                            max_temp=20.0, min_temp=15.0, precipitation_sum=1.0,
                            wind_speed=15.0, wind_gusts=25.0, cloud_cover=40.0, humidity=60.0, pressure_msl=1015.0,
                            raw_json={},
                        ),
                    ],
                )

    result = compute_data_quality(db_path, [LOCATION.name, other.name], window_days=30)
    assert result["Melbourne"]["missing_pct"] == 0.0
    assert result["Sydney"]["missing_pct"] > 0.0
    # Pooled must reflect Sydney's gap even though Melbourne is perfect.
    assert result["__ALL__"]["missing_pct"] > 0.0
    assert result["__ALL__"]["missing_pct"] < result["Sydney"]["missing_pct"]  # diluted by Melbourne's clean record
