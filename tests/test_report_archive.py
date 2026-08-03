from __future__ import annotations

from datetime import UTC, datetime, timedelta

from weather_ensemble.report_archive import (
    RETENTION_DAYS,
    list_report_snapshots,
    load_report_snapshot,
    save_report_snapshot,
)

NOW = datetime(2026, 8, 4, 5, 0, 0, tzinfo=UTC).replace(tzinfo=None)


def test_save_and_load_report_snapshot_round_trips(tmp_path):
    db_path = tmp_path / "weather.db"
    html = "<html><body>report contents</body></html>"

    result = save_report_snapshot(db_path, html, generated_at=NOW)

    assert result["generated_at"] == NOW.isoformat(timespec="seconds")
    assert result["stored_bytes"] > 0
    assert result["pruned"] == 0

    loaded = load_report_snapshot(db_path, NOW.isoformat(timespec="seconds"))
    assert loaded == html


def test_load_report_snapshot_returns_none_for_unknown_timestamp(tmp_path):
    db_path = tmp_path / "weather.db"
    assert load_report_snapshot(db_path, "2020-01-01T00:00:00") is None


def test_list_report_snapshots_newest_first(tmp_path):
    db_path = tmp_path / "weather.db"
    day1 = NOW - timedelta(days=2)
    day2 = NOW - timedelta(days=1)
    day3 = NOW

    for when in (day1, day3, day2):  # insert out of order
        save_report_snapshot(db_path, f"<html>{when.isoformat()}</html>", generated_at=when)

    timestamps = list_report_snapshots(db_path)

    assert timestamps == [
        day3.isoformat(timespec="seconds"),
        day2.isoformat(timespec="seconds"),
        day1.isoformat(timespec="seconds"),
    ]


def test_save_report_snapshot_prunes_older_than_retention(tmp_path):
    """A snapshot from beyond RETENTION_DAYS ago must disappear the next
    time a new one is saved - pruning is self-cleaning, not a separate
    maintenance step someone has to remember to run.
    """
    db_path = tmp_path / "weather.db"
    old = NOW - timedelta(days=RETENTION_DAYS + 5)
    recent = NOW - timedelta(days=RETENTION_DAYS - 5)

    save_report_snapshot(db_path, "<html>old</html>", generated_at=old)
    result = save_report_snapshot(db_path, "<html>recent</html>", generated_at=recent)
    assert result["pruned"] == 0  # nothing to prune yet relative to `recent`'s own cutoff

    result = save_report_snapshot(db_path, "<html>now</html>", generated_at=NOW)

    assert result["pruned"] == 1  # `old` is now beyond RETENTION_DAYS relative to NOW
    timestamps = list_report_snapshots(db_path)
    assert old.isoformat(timespec="seconds") not in timestamps
    assert recent.isoformat(timespec="seconds") in timestamps
    assert NOW.isoformat(timespec="seconds") in timestamps
