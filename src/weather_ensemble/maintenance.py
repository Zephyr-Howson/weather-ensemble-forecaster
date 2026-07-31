from __future__ import annotations

from pathlib import Path
from typing import Any

from weather_ensemble import db
from weather_ensemble.config import get_periods_db_path

# For each table: the columns that identify "the same prediction/observation",
# and the ORDER BY that ranks duplicates newest-first so ROW_NUMBER() = 1 is
# the one kept. `forecasts` intentionally does NOT rank by collected_at alone:
# a live-collected row always outranks a backfilled one for the same
# (source, location, forecast_date), even if the backfill happened later -
# this matches the precedence load_modelling_table already uses everywhere
# else in the app (backfill is a lower-priority historical reconstruction,
# not "fresher" data). Everything else has no such distinction, so newest
# generated_at/collected_at wins outright.
_DEDUPE_SPECS = {
    "forecasts": {
        "partition": ["source", "location_name", "forecast_date"],
        "order": "CASE WHEN collection_method = 'live' THEN 0 ELSE 1 END, collected_at DESC",
    },
    "actuals": {
        "partition": ["source", "location_name", "actual_date"],
        "order": "collected_at DESC",
    },
    "ensemble_predictions": {
        "partition": ["location_name", "forecast_date"],
        "order": "generated_at DESC",
    },
    "ml_predictions": {
        "partition": ["location_name", "forecast_date"],
        "order": "generated_at DESC",
    },
    "best_predictions": {
        "partition": ["location_name", "forecast_date"],
        "order": "generated_at DESC",
    },
}

# Same idea, one level finer - every period table's identity additionally
# includes `period`, since a row is "the same prediction/observation" only if
# it's for the same sub-daily bucket too.
_PERIOD_DEDUPE_SPECS = {
    "forecast_periods": {
        "partition": ["source", "location_name", "forecast_date", "period"],
        "order": "CASE WHEN collection_method = 'live' THEN 0 ELSE 1 END, collected_at DESC",
    },
    "actual_periods": {
        "partition": ["source", "location_name", "actual_date", "period"],
        "order": "collected_at DESC",
    },
    "ensemble_predictions_periods": {
        "partition": ["location_name", "forecast_date", "period"],
        "order": "generated_at DESC",
    },
    "ml_predictions_periods": {
        "partition": ["location_name", "forecast_date", "period"],
        "order": "generated_at DESC",
    },
    "best_predictions_periods": {
        "partition": ["location_name", "forecast_date", "period"],
        "order": "generated_at DESC",
    },
}


def _dedupe_tables(conn, specs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for table, spec in specs.items():
        partition_cols = ", ".join(spec["partition"])
        before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.execute(
            f"""
            DELETE FROM {table}
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY {partition_cols} ORDER BY {spec["order"]}
                    ) AS rn
                    FROM {table}
                ) WHERE rn > 1
            )
            """
        )
        after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        report[table] = {"rows_before": before, "rows_after": after, "removed": before - after}
    return report


def deduplicate(db_path: Path) -> dict[str, Any]:
    """Remove duplicate rows (same source/location/date[/period]) across every
    prediction/observation table in both databases, keeping only the
    highest-priority one per the rules above.

    Safe to run repeatedly - a table with no duplicates reports 0 removed.
    """
    report: dict[str, Any] = {}
    with db.connect(db_path) as conn:
        report.update(_dedupe_tables(conn, _DEDUPE_SPECS))
        conn.commit()
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        report.update(_dedupe_tables(conn, _PERIOD_DEDUPE_SPECS))
        conn.commit()
    return report
