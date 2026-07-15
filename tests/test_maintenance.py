from __future__ import annotations

import json

from weather_ensemble.db import connect
from weather_ensemble.maintenance import deduplicate


def _insert_forecast(conn, source, location, forecast_date, collected_at, collection_method, max_temp):
    conn.execute(
        """
        INSERT INTO forecasts (
            source, location_name, lat, lon, forecast_date, collected_at,
            max_temp, collection_method, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (source, location, -37.8, 144.9, forecast_date, collected_at, max_temp, collection_method, "{}"),
    )


def _insert_ensemble(conn, location, forecast_date, generated_at, max_temp):
    conn.execute(
        """
        INSERT INTO ensemble_predictions (
            location_name, lat, lon, forecast_date, generated_at, window_days, max_temp, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (location, -37.8, 144.9, forecast_date, generated_at, 30, max_temp, json.dumps({})),
    )


def test_dedupe_forecasts_prefers_live_over_backfill_even_if_older(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_forecast(conn, "open_meteo_gfs_global", "Melbourne", "2026-07-14", "2026-07-13T11:11:36", "live", 14.9)
        _insert_forecast(conn, "open_meteo_gfs_global", "Melbourne", "2026-07-14", "2026-07-13T21:00:00", "backfill", 14.2)

    report = deduplicate(db_path)
    assert report["forecasts"]["removed"] == 1

    with connect(db_path) as conn:
        rows = conn.execute("SELECT collection_method, max_temp FROM forecasts").fetchall()
    assert len(rows) == 1
    assert rows[0]["collection_method"] == "live"
    assert rows[0]["max_temp"] == 14.9


def test_dedupe_ensemble_predictions_keeps_newest_generated_at(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_ensemble(conn, "Melbourne", "2026-07-15", "2026-07-14T09:58:59", 14.0)
        _insert_ensemble(conn, "Melbourne", "2026-07-15", "2026-07-14T21:24:10", 14.58)

    report = deduplicate(db_path)
    assert report["ensemble_predictions"]["removed"] == 1

    with connect(db_path) as conn:
        rows = conn.execute("SELECT generated_at, max_temp FROM ensemble_predictions").fetchall()
    assert len(rows) == 1
    assert rows[0]["generated_at"] == "2026-07-14T21:24:10"
    assert rows[0]["max_temp"] == 14.58


def test_dedupe_is_idempotent_with_no_duplicates(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_forecast(conn, "open_meteo_gfs_global", "Melbourne", "2026-07-14", "2026-07-13T11:11:36", "live", 14.9)

    report = deduplicate(db_path)
    assert report["forecasts"]["removed"] == 0
    report_again = deduplicate(db_path)
    assert report_again["forecasts"]["removed"] == 0
