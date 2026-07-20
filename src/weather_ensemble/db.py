from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from weather_ensemble.models import ActualRecord, ForecastRecord

FORECAST_COLUMNS = [
    "source", "location_name", "lat", "lon", "forecast_date", "collected_at",
    "max_temp", "min_temp", "rain_probability", "precipitation_sum",
    "wind_speed", "wind_gusts", "cloud_cover", "humidity", "pressure_msl",
    "weather_code", "raw_json", "collection_method",
]

ACTUAL_COLUMNS = [
    "source", "location_name", "lat", "lon", "actual_date", "collected_at",
    "max_temp", "min_temp", "precipitation_sum", "did_rain",
    "wind_speed", "wind_gusts", "cloud_cover", "humidity", "pressure_msl",
    "weather_code", "raw_json",
]


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _drop_column_if_present(conn: sqlite3.Connection, table: str, column: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in existing:
        conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            location_name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            forecast_date TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            max_temp REAL,
            min_temp REAL,
            rain_probability REAL,
            precipitation_sum REAL,
            wind_speed REAL,
            wind_gusts REAL,
            cloud_cover REAL,
            humidity REAL,
            pressure_msl REAL,
            weather_code REAL,
            raw_json TEXT,
            collection_method TEXT,
            UNIQUE(source, location_name, forecast_date, collected_at)
        );

        CREATE INDEX IF NOT EXISTS idx_forecasts_lookup
            ON forecasts(location_name, forecast_date, source);

        CREATE TABLE IF NOT EXISTS actuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            location_name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            actual_date TEXT NOT NULL,
            collected_at TEXT NOT NULL,
            max_temp REAL,
            min_temp REAL,
            precipitation_sum REAL,
            did_rain INTEGER,
            wind_speed REAL,
            wind_gusts REAL,
            cloud_cover REAL,
            humidity REAL,
            pressure_msl REAL,
            weather_code REAL,
            raw_json TEXT,
            UNIQUE(source, location_name, actual_date)
        );

        CREATE INDEX IF NOT EXISTS idx_actuals_lookup
            ON actuals(location_name, actual_date, source);

        CREATE TABLE IF NOT EXISTS ensemble_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            forecast_date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            window_days INTEGER NOT NULL,
            max_temp REAL,
            min_temp REAL,
            rain_probability REAL,
            precipitation_sum REAL,
            did_rain REAL,
            wind_speed REAL,
            wind_gusts REAL,
            cloud_cover REAL,
            humidity REAL,
            pressure_msl REAL,
            weather_code REAL,
            metadata_json TEXT,
            UNIQUE(location_name, forecast_date, generated_at)
        );

        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            forecast_date TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            max_temp REAL,
            min_temp REAL,
            precipitation_sum REAL,
            did_rain REAL,
            did_rain_probability REAL,
            wind_speed REAL,
            wind_gusts REAL,
            cloud_cover REAL,
            humidity REAL,
            pressure_msl REAL,
            metadata_json TEXT,
            UNIQUE(location_name, forecast_date, generated_at)
        );
        """
    )

    # Lightweight migrations for existing local databases created by earlier versions.
    for col in ["precipitation_sum", "wind_gusts", "cloud_cover", "humidity", "pressure_msl", "weather_code"]:
        _add_column_if_missing(conn, "forecasts", col, "REAL")
    _add_column_if_missing(conn, "forecasts", "collection_method", "TEXT")
    for col in ["did_rain", "wind_gusts", "cloud_cover", "humidity", "pressure_msl", "weather_code"]:
        _add_column_if_missing(conn, "actuals", col, "REAL")
    # Earlier versions had actuals.rain_probability. Leave it if present; no new code uses it.
    for col in ["precipitation_sum", "did_rain", "wind_gusts", "cloud_cover", "humidity", "pressure_msl", "weather_code"]:
        _add_column_if_missing(conn, "ensemble_predictions", col, "REAL")
    for col in ["cloud_cover", "humidity", "pressure_msl"]:
        _add_column_if_missing(conn, "ml_predictions", col, "REAL")

    # uv_index was removed: forecast and actual sources turned out to measure
    # different things (forecast-side values ran ~2-4x the observed ground
    # truth, a systematic bias not noise - see the "why is the UV trend graph
    # strange" investigation), making it meaningless to keep. Drop it from any
    # database created by an earlier version.
    for table in ["forecasts", "actuals", "ensemble_predictions", "ml_predictions"]:
        _drop_column_if_present(conn, table, "uv_index")

    # Rows inserted before collection_method existed have no way to record how
    # they were collected. Recover it from the raw_json tag that
    # fetch_historical_forecasts stamps on backfilled rows; everything else was
    # collected live.
    conn.execute(
        "UPDATE forecasts SET collection_method = 'backfill' "
        "WHERE collection_method IS NULL AND raw_json LIKE '%historical_forecast_or_past_days%'"
    )
    conn.execute("UPDATE forecasts SET collection_method = 'live' WHERE collection_method IS NULL")
    conn.commit()


def insert_forecasts(conn: sqlite3.Connection, records: Iterable[ForecastRecord]) -> int:
    inserted = 0
    placeholders = ", ".join("?" for _ in FORECAST_COLUMNS)
    column_list = ", ".join(FORECAST_COLUMNS)
    for r in records:
        values = (
            r.source, r.location_name, r.lat, r.lon, r.forecast_date.isoformat(),
            r.collected_at.isoformat(timespec="seconds"), r.max_temp, r.min_temp,
            r.rain_probability, r.precipitation_sum, r.wind_speed,
            r.wind_gusts, r.cloud_cover, r.humidity, r.pressure_msl, r.weather_code,
            json.dumps(r.raw_json or {}), r.collection_method,
        )
        cur = conn.execute(
            f"INSERT OR IGNORE INTO forecasts ({column_list}) VALUES ({placeholders})",
            values,
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def upsert_actual(conn: sqlite3.Connection, r: ActualRecord) -> None:
    conn.execute(
        """
        INSERT INTO actuals (
            source, location_name, lat, lon, actual_date, collected_at,
            max_temp, min_temp, precipitation_sum, did_rain,
            wind_speed, wind_gusts, cloud_cover, humidity, pressure_msl,
            weather_code, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, location_name, actual_date)
        DO UPDATE SET
            collected_at=excluded.collected_at,
            max_temp=excluded.max_temp,
            min_temp=excluded.min_temp,
            precipitation_sum=excluded.precipitation_sum,
            did_rain=excluded.did_rain,
            wind_speed=excluded.wind_speed,
            wind_gusts=excluded.wind_gusts,
            cloud_cover=excluded.cloud_cover,
            humidity=excluded.humidity,
            pressure_msl=excluded.pressure_msl,
            weather_code=excluded.weather_code,
            raw_json=excluded.raw_json
        """,
        (
            r.source, r.location_name, r.lat, r.lon, r.actual_date.isoformat(),
            r.collected_at.isoformat(timespec="seconds"), r.max_temp, r.min_temp,
            r.precipitation_sum, r.did_rain, r.wind_speed, r.wind_gusts,
            r.cloud_cover, r.humidity, r.pressure_msl, r.weather_code,
            json.dumps(r.raw_json or {}),
        ),
    )
    conn.commit()
