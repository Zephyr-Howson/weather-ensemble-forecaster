from __future__ import annotations

import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from weather_ensemble import db
from weather_ensemble.config import get_report_archive_db_path

# Snapshots older than this are dropped, not thinned to a lower frequency.
# Each gzipped self-contained HTML report (embedded Plotly data) is around
# 1 MB, and the archive DB is committed to git by the
# regenerate-report workflow, so it must not grow without bound.
RETENTION_DAYS = 30


def save_report_snapshot(db_path: Path, html: str, generated_at: datetime | None = None) -> dict[str, Any]:
    """Gzip-compress the just-rendered report and store it as one row of
    report_snapshots (in the separate archive DB, see
    get_report_archive_db_path), keyed by generated_at, then delete every
    snapshot older than RETENTION_DAYS before generated_at.

    `generated_at` defaults to the current UTC time (naive, to the second); a
    second save with the same timestamp is ignored. Pruning runs on every save,
    so the archive needs no separate maintenance step. Returns the stored
    timestamp, the compressed size in bytes and the number of rows pruned.
    """
    generated_at = generated_at or datetime.now(UTC).replace(tzinfo=None)
    generated_at_iso = generated_at.isoformat(timespec="seconds")
    compressed = gzip.compress(html.encode("utf-8"))
    cutoff = (generated_at - timedelta(days=RETENTION_DAYS)).isoformat(timespec="seconds")

    with db.connect_report_archive(get_report_archive_db_path(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO report_snapshots (generated_at, html_gzip) VALUES (?, ?)",
            (generated_at_iso, compressed),
        )
        deleted = conn.execute("DELETE FROM report_snapshots WHERE generated_at < ?", (cutoff,)).rowcount
        conn.commit()

    return {
        "generated_at": generated_at_iso,
        "stored_bytes": len(compressed),
        "pruned": deleted,
    }


def list_report_snapshots(db_path: Path) -> list[str]:
    """Every archived snapshot's generated_at timestamp, newest first."""
    with db.connect_report_archive(get_report_archive_db_path(db_path)) as conn:
        rows = conn.execute("SELECT generated_at FROM report_snapshots ORDER BY generated_at DESC").fetchall()
    return [row["generated_at"] for row in rows]


def load_report_snapshot(db_path: Path, generated_at: str) -> str | None:
    """The decompressed HTML for one archived snapshot, or None if that
    exact generated_at timestamp isn't in the archive (already pruned, or
    never existed) - see list_report_snapshots for the available values.
    """
    with db.connect_report_archive(get_report_archive_db_path(db_path)) as conn:
        row = conn.execute(
            "SELECT html_gzip FROM report_snapshots WHERE generated_at = ?", (generated_at,)
        ).fetchone()
    return gzip.decompress(row["html_gzip"]).decode("utf-8") if row is not None else None
