from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from weather_ensemble.models import ActualRecord, ForecastRecord


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


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
            uv_index REAL,
            wind_speed REAL,
            raw_json TEXT,
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
            rain_probability REAL,
            precipitation_sum REAL,
            uv_index REAL,
            wind_speed REAL,
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
            uv_index REAL,
            wind_speed REAL,
            metadata_json TEXT,
            UNIQUE(location_name, forecast_date, generated_at)
        );
        """
    )
    conn.commit()


def insert_forecasts(conn: sqlite3.Connection, records: Iterable[ForecastRecord]) -> int:
    inserted = 0
    for r in records:
        cur = conn.execute(
            """
            INSERT OR IGNORE INTO forecasts (
                source, location_name, lat, lon, forecast_date, collected_at,
                max_temp, min_temp, rain_probability, uv_index, wind_speed, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r.source,
                r.location_name,
                r.lat,
                r.lon,
                r.forecast_date.isoformat(),
                r.collected_at.isoformat(timespec="seconds"),
                r.max_temp,
                r.min_temp,
                r.rain_probability,
                r.uv_index,
                r.wind_speed,
                json.dumps(r.raw_json or {}),
            ),
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def upsert_actual(conn: sqlite3.Connection, r: ActualRecord) -> None:
    conn.execute(
        """
        INSERT INTO actuals (
            source, location_name, lat, lon, actual_date, collected_at,
            max_temp, min_temp, rain_probability, precipitation_sum,
            uv_index, wind_speed, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, location_name, actual_date)
        DO UPDATE SET
            collected_at=excluded.collected_at,
            max_temp=excluded.max_temp,
            min_temp=excluded.min_temp,
            rain_probability=excluded.rain_probability,
            precipitation_sum=excluded.precipitation_sum,
            uv_index=excluded.uv_index,
            wind_speed=excluded.wind_speed,
            raw_json=excluded.raw_json
        """,
        (
            r.source,
            r.location_name,
            r.lat,
            r.lon,
            r.actual_date.isoformat(),
            r.collected_at.isoformat(timespec="seconds"),
            r.max_temp,
            r.min_temp,
            r.rain_probability,
            r.precipitation_sum,
            r.uv_index,
            r.wind_speed,
            json.dumps(r.raw_json or {}),
        ),
    )
    conn.commit()
