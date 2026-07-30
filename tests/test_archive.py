from __future__ import annotations

import json

import pandas as pd

from weather_ensemble.archive import archive_old_forecasts, export_blob_backlog
from weather_ensemble.config import get_periods_db_path
from weather_ensemble.db import connect, connect_periods


def _insert_forecast(conn, forecast_date, collected_at, raw_json="{}", max_temp=20.0):
    conn.execute(
        """
        INSERT INTO forecasts (
            source, location_name, lat, lon, forecast_date, collected_at, max_temp, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("open_meteo_gfs_global", "Melbourne", -37.8, 144.9, forecast_date, collected_at, max_temp, raw_json),
    )


def _insert_ensemble(conn, forecast_date, generated_at, metadata_json="{}"):
    conn.execute(
        """
        INSERT INTO ensemble_predictions (
            location_name, lat, lon, forecast_date, generated_at, window_days, max_temp, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("Melbourne", -37.8, 144.9, forecast_date, generated_at, 30, 14.0, metadata_json),
    )


def test_export_blob_backlog_moves_raw_json_to_parquet_and_clears_it_live(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_forecast(conn, "2026-07-14", "2026-07-13T11:11:36", raw_json=json.dumps({"hourly": [1, 2, 3]}))
        conn.commit()

    report = export_blob_backlog(db_path)
    assert report["forecasts"]["archived"] == 1

    archive_path = tmp_path / "archive" / "forecasts_raw_json.parquet"
    assert archive_path.exists()
    archived = pd.read_parquet(archive_path)
    assert len(archived) == 1
    assert json.loads(archived.iloc[0]["raw_json"]) == {"hourly": [1, 2, 3]}

    with connect(db_path) as conn:
        row = conn.execute("SELECT raw_json, max_temp FROM forecasts").fetchone()
    assert row["raw_json"] is None
    assert row["max_temp"] == 20.0  # every other column is untouched


def test_export_blob_backlog_is_idempotent(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_forecast(conn, "2026-07-14", "2026-07-13T11:11:36", raw_json="{}")
        conn.commit()

    first = export_blob_backlog(db_path)
    assert first["forecasts"]["archived"] == 1
    second = export_blob_backlog(db_path)
    assert second["forecasts"]["archived"] == 0


def test_export_blob_backlog_covers_ensemble_metadata_and_period_tables(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_ensemble(conn, "2026-07-15", "2026-07-14T09:58:59", metadata_json=json.dumps({"scores": {"a": 1.0}}))
        conn.commit()
    with connect_periods(get_periods_db_path(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO ensemble_predictions_periods (
                location_name, lat, lon, forecast_date, period, generated_at, window_days, precipitation_sum, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("Melbourne", -37.8, 144.9, "2026-07-15", "morning", "2026-07-14T09:58:59", 30, 0.5, json.dumps({"x": 1})),
        )
        conn.commit()

    report = export_blob_backlog(db_path)
    assert report["ensemble_predictions"]["archived"] == 1
    assert report["ensemble_predictions_periods"]["archived"] == 1

    with connect(db_path) as conn:
        row = conn.execute("SELECT metadata_json FROM ensemble_predictions").fetchone()
    assert row["metadata_json"] is None
    with connect_periods(get_periods_db_path(db_path)) as conn:
        row = conn.execute("SELECT metadata_json FROM ensemble_predictions_periods").fetchone()
    assert row["metadata_json"] is None


def test_archive_old_forecasts_removes_only_rows_past_the_cutoff(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_forecast(conn, "2025-01-01", "2024-12-31T21:00:00", max_temp=15.0)  # ~200 days before test's "now"
        _insert_forecast(conn, "2026-07-20", "2026-07-19T21:00:00", max_temp=22.0)  # recent
        conn.commit()

    report = archive_old_forecasts(db_path, cutoff_days=180)
    assert report["forecasts"]["archived"] == 1

    with connect(db_path) as conn:
        rows = conn.execute("SELECT forecast_date, max_temp FROM forecasts").fetchall()
    assert len(rows) == 1
    assert rows[0]["forecast_date"] == "2026-07-20"

    archive_path = tmp_path / "archive" / "forecasts_history.parquet"
    archived = pd.read_parquet(archive_path)
    assert len(archived) == 1
    assert archived.iloc[0]["forecast_date"] == "2025-01-01"
    assert archived.iloc[0]["max_temp"] == 15.0


def test_archive_old_forecasts_appends_across_repeated_runs_without_duplicating(tmp_path):
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_forecast(conn, "2025-01-01", "2024-12-31T21:00:00", max_temp=15.0)
        conn.commit()

    first = archive_old_forecasts(db_path, cutoff_days=180)
    assert first["forecasts"]["archived"] == 1

    with connect(db_path) as conn:
        _insert_forecast(conn, "2025-02-01", "2025-01-31T21:00:00", max_temp=16.0)
        conn.commit()

    second = archive_old_forecasts(db_path, cutoff_days=180)
    assert second["forecasts"]["archived"] == 1

    archive_path = tmp_path / "archive" / "forecasts_history.parquet"
    archived = pd.read_parquet(archive_path)
    assert len(archived) == 2
    assert set(archived["forecast_date"]) == {"2025-01-01", "2025-02-01"}


def test_archive_old_forecasts_never_touches_actuals_or_predictions(tmp_path):
    """Actuals and every prediction table are kept forever per the retention
    policy - only the raw forecasts table (and its period equivalent) ages
    out. Seed an old actual/ensemble-prediction row alongside an old forecast
    row and confirm only the forecast row is removed.
    """
    db_path = tmp_path / "weather.db"
    with connect(db_path) as conn:
        _insert_forecast(conn, "2025-01-01", "2024-12-31T21:00:00")
        conn.execute(
            "INSERT INTO actuals (source, location_name, lat, lon, actual_date, collected_at, max_temp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("open_meteo_archive", "Melbourne", -37.8, 144.9, "2025-01-01", "2025-01-01T00:00:00", 15.0),
        )
        _insert_ensemble(conn, "2025-01-01", "2024-12-31T21:00:00")
        conn.commit()

    archive_old_forecasts(db_path, cutoff_days=180)

    with connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM actuals").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM ensemble_predictions").fetchone()[0] == 1
