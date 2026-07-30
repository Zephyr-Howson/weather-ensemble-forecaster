from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import get_periods_db_path

# Cold storage lives alongside the two live databases, git-committed like
# them. Parquet (columnar + compressed) rather than another SQLite file:
# this data is only ever bulk-read on demand (pd.read_parquet/DuckDB), never
# joined live against the app's own tables, and compresses far better than
# SQLite for the mostly-repetitive JSON/text it holds.


def _archive_dir(db_path: Path) -> Path:
    return db_path.parent / "archive"


def _merge_into_parquet(new_rows: pd.DataFrame, archive_path: Path, dedupe_col: str = "id") -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        existing = pd.read_parquet(archive_path)
        combined = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates(subset=[dedupe_col], keep="last")
    else:
        combined = new_rows
    combined.to_parquet(archive_path, index=False)


# For each write-only JSON blob column - written on every insert, never read
# back by report.py/scoring.py/ml.py except one narrow legacy migration check
# in db.py - the key columns needed to identify the row later, keyed by which
# database the table lives in. Found via a size audit: these six columns
# accounted for ~78MB of a ~90MB database. Exported once to Parquet, then
# cleared from the live row - see export_blob_backlog.
_MAIN_BLOB_SPECS = [
    ("forecasts", "raw_json", ["id", "source", "location_name", "forecast_date", "collected_at"]),
    ("actuals", "raw_json", ["id", "source", "location_name", "actual_date", "collected_at"]),
    ("ensemble_predictions", "metadata_json", ["id", "location_name", "forecast_date", "generated_at"]),
    ("ml_predictions", "metadata_json", ["id", "location_name", "forecast_date", "generated_at"]),
]
_PERIOD_BLOB_SPECS = [
    ("ensemble_predictions_periods", "metadata_json", ["id", "location_name", "forecast_date", "period", "generated_at"]),
    ("ml_predictions_periods", "metadata_json", ["id", "location_name", "forecast_date", "period", "generated_at"]),
]


def _archive_and_clear_blob_column(
    conn, table: str, blob_col: str, key_columns: list[str], archive_path: Path
) -> dict[str, Any]:
    cols = ", ".join([*key_columns, blob_col])
    df = pd.read_sql_query(f"SELECT {cols} FROM {table} WHERE {blob_col} IS NOT NULL", conn)
    if df.empty:
        return {"archived": 0, "path": str(archive_path)}

    _merge_into_parquet(df, archive_path)
    conn.execute(f"UPDATE {table} SET {blob_col} = NULL WHERE {blob_col} IS NOT NULL")
    return {"archived": len(df), "path": str(archive_path)}


def export_blob_backlog(db_path: Path) -> dict[str, Any]:
    """Move every write-only JSON blob column's existing content to Parquet,
    then clear it from the live row - see _MAIN_BLOB_SPECS/_PERIOD_BLOB_SPECS.

    Safe to run repeatedly: once a row's blob is cleared there's nothing left
    to re-export, so a second run reports 0 archived everywhere. Not part of
    the daily pipeline - the code stops writing these columns going forward
    (see service.py/ml.py/backtest.py), so there's no new backlog to sweep;
    this only ever needs to run once per existing database.
    """
    archive_dir = _archive_dir(db_path)
    report: dict[str, Any] = {}

    with db.connect(db_path) as conn:
        for table, blob_col, key_cols in _MAIN_BLOB_SPECS:
            report[table] = _archive_and_clear_blob_column(conn, table, blob_col, key_cols, archive_dir / f"{table}_{blob_col}.parquet")
        conn.commit()

    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        for table, blob_col, key_cols in _PERIOD_BLOB_SPECS:
            report[table] = _archive_and_clear_blob_column(conn, table, blob_col, key_cols, archive_dir / f"{table}_{blob_col}.parquet")
        conn.commit()

    return report


def _archive_and_delete_old_rows(conn, table: str, date_col: str, cutoff_iso: str, archive_path: Path) -> dict[str, Any]:
    df = pd.read_sql_query(f"SELECT * FROM {table} WHERE {date_col} < ?", conn, params=(cutoff_iso,))
    if df.empty:
        return {"archived": 0, "path": str(archive_path)}

    _merge_into_parquet(df, archive_path)
    conn.execute(f"DELETE FROM {table} WHERE {date_col} < ?", (cutoff_iso,))
    return {"archived": len(df), "path": str(archive_path)}


def archive_old_forecasts(db_path: Path, cutoff_days: int = 180) -> dict[str, Any]:
    """Move forecasts/forecast_periods rows older than cutoff_days to Parquet,
    then delete them from the live database.

    Actuals and every prediction table (ensemble/ML, daily and per-period)
    are deliberately untouched here and kept forever - only raw forecasts age
    out of the live database, since they're the one table whose row count
    grows with the number of raw sources rather than just time, and they're
    fully recoverable from the archive if ever needed again.

    Safe to run repeatedly/on a schedule: once a row is archived and deleted
    it can't be re-selected, so re-running finds nothing new until the next
    batch of rows crosses the cutoff.
    """
    cutoff_iso = (datetime.now(UTC).date() - timedelta(days=cutoff_days)).isoformat()
    archive_dir = _archive_dir(db_path)
    report: dict[str, Any] = {}

    with db.connect(db_path) as conn:
        report["forecasts"] = _archive_and_delete_old_rows(
            conn, "forecasts", "forecast_date", cutoff_iso, archive_dir / "forecasts_history.parquet"
        )
        conn.commit()

    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        report["forecast_periods"] = _archive_and_delete_old_rows(
            conn, "forecast_periods", "forecast_date", cutoff_iso, archive_dir / "forecast_periods_history.parquet"
        )
        conn.commit()

    return report
