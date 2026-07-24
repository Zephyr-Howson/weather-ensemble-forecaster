from __future__ import annotations

from datetime import UTC, date, datetime

from weather_ensemble.config import Location
from weather_ensemble.db import connect, insert_forecasts, upsert_actual
from weather_ensemble.models import ActualRecord, ForecastRecord
from weather_ensemble.service import blend_forecast

LOCATION = Location(name="Melbourne", lat=-37.8136, lon=144.9631, timezone="Australia/Melbourne")


def _forecast(source: str, forecast_date: date, value: float) -> ForecastRecord:
    return ForecastRecord(
        source=source,
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        forecast_date=forecast_date,
        collected_at=datetime.now(UTC).replace(microsecond=0, tzinfo=None),
        max_temp=value,
        raw_json={},
    )


def _actual(actual_date: date, value: float) -> ActualRecord:
    return ActualRecord(
        source="open_meteo_archive",
        location_name=LOCATION.name,
        lat=LOCATION.lat,
        lon=LOCATION.lon,
        actual_date=actual_date,
        collected_at=datetime.now(UTC).replace(microsecond=0, tzinfo=None),
        max_temp=value,
        raw_json={},
    )


def test_blend_forecast_honors_explicit_target_date(tmp_path):
    """target_date exists to reconstruct a specific past day's prediction (e.g. one
    dropped by a database reset) rather than always defaulting to tomorrow."""
    db_path = tmp_path / "weather.db"
    a_past_date = date(2026, 6, 1)

    with connect(db_path) as conn:
        insert_forecasts(conn, [_forecast("open_meteo_best_match", a_past_date, 18.0)])

    result = blend_forecast(db_path, LOCATION, window_days=3650, target_date=a_past_date)

    assert result["forecast_date"] == a_past_date.isoformat()
    assert result["blended"]["max_temp"] == 18.0


def test_blend_forecast_reconstruction_excludes_lookahead_history(tmp_path):
    """MAE weighting for a reconstructed target_date must only use data strictly
    before it - the same no-lookahead rule the walk-forward backtest enforces.
    If a source's *later* accuracy were allowed to influence its weight for an
    earlier target_date, that's leaking data the real pipeline wouldn't have had
    yet at that point in time.

    good_source is accurate before target_date and inaccurate after; the reverse
    is true for late_bloomer. Without the target_date < filter, late_bloomer's
    good post-target_date performance would pull its weight up and drag the
    blended value toward its forecast (11.0); with the filter, good_source's
    real historical edge dominates and the blend stays close to good_source's
    forecast (21.0).
    """
    db_path = tmp_path / "weather.db"
    target = date(2026, 6, 10)
    before1, before2, after = date(2026, 6, 8), date(2026, 6, 9), date(2026, 6, 11)
    actual_value = 20.0

    with connect(db_path) as conn:
        insert_forecasts(
            conn,
            [
                _forecast("good_source", before1, 20.1),  # error 0.1
                _forecast("good_source", before2, 20.1),  # error 0.1
                _forecast("good_source", after, 30.0),  # error 10 (would only matter if leaked)
                _forecast("good_source", target, 21.0),  # the actual prediction being blended
                _forecast("late_bloomer", before1, 10.0),  # error 10
                _forecast("late_bloomer", before2, 10.0),  # error 10
                _forecast("late_bloomer", after, 20.0),  # error 0 (would only matter if leaked)
                _forecast("late_bloomer", target, 11.0),  # the actual prediction being blended
            ],
        )
        for d in (before1, before2, after):
            upsert_actual(conn, _actual(d, actual_value))

    result = blend_forecast(db_path, LOCATION, window_days=3650, target_date=target)

    # If the post-target_date day leaked into the weighting, the blend would
    # fall closer to late_bloomer's forecast (11.0), well below this threshold.
    assert result["blended"]["max_temp"] > 19.0
