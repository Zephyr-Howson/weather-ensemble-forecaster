from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from weather_ensemble import db
from weather_ensemble.config import FORECAST_VARIABLES
from weather_ensemble.scoring import MODEL_BEST, MODEL_ENSEMBLE, MODEL_ML

# Every raw forecast variable a source can report (see FORECAST_VARIABLES).
_RAW_FIELDS = list(FORECAST_VARIABLES)

# Each meta-model's own field set, mapping the shared "did_rain" concept to
# whichever probability column that model actually stores - mirrors how
# report._recent_forecast_data already reads these same three tables.
_META_TABLES = {MODEL_ENSEMBLE: "ensemble_predictions", MODEL_ML: "ml_predictions", MODEL_BEST: "best_predictions"}
_META_FIELDS = {
    MODEL_ENSEMBLE: ["max_temp", "min_temp", "precipitation_sum", "rain_probability", "wind_speed", "wind_gusts", "cloud_cover", "humidity", "pressure_msl"],
    MODEL_ML: ["max_temp", "min_temp", "precipitation_sum", "did_rain_probability", "wind_speed", "wind_gusts", "cloud_cover", "humidity", "pressure_msl"],
    MODEL_BEST: ["max_temp", "min_temp", "precipitation_sum", "did_rain_probability", "wind_speed", "wind_gusts", "cloud_cover", "humidity", "pressure_msl"],
}


def _pct(n: int, total: int) -> float:
    return round(n / total * 100, 2) if total else 0.0


def _entity_metrics_for_location(
    existing_by_date: dict, capability: set[str], fields: list[str], first_seen, window_start: pd.Timestamp, anchor: pd.Timestamp
) -> dict[str, int]:
    """Shared per-(entity, location) tally: missing calendar days against
    that entity's own first-ever appearance (never penalizing a source/model
    for days before it started existing - e.g. a live-only provider added
    partway through history), plus null counts restricted to fields this
    entity has ever actually reported (so a source that structurally never
    covers a field - e.g. BOM has no wind numerics - never inflates the
    "unexpected null" count; only genuine gaps in a field it does cover do).
    """
    if first_seen is None:
        return {"missing": 0, "missing_total": 0, "null": 0, "null_total": 0}

    expected_start = max(first_seen, window_start)
    expected_dates = pd.date_range(expected_start, anchor, freq="D")
    missing = sum(1 for d in expected_dates if d not in existing_by_date)
    missing_total = len(expected_dates)

    null_count = 0
    null_total = 0
    for field in capability & set(fields):
        for row in existing_by_date.values():
            null_total += 1
            if pd.isna(row.get(field)):
                null_count += 1

    return {"missing": missing, "missing_total": missing_total, "null": null_count, "null_total": null_total}


def compute_data_quality(db_path: Path, location_names: list[str], window_days: int = 30) -> dict[str, Any]:
    """Missing-row and unexpected-null rates over the trailing window_days,
    for every raw forecast source plus the three meta-models (Weighted/ML/
    Best), both pooled across `location_names` and broken out per location.

    Takes plain location names rather than `Location` objects - nothing
    here needs lat/lon, and report.py (the caller) only has names on hand
    at this point anyway, derived from the already-scored long_df.

    Two distinct failure modes, both real (see the investigation that
    prompted this): a "missing row" is a source/model having produced no
    forecast at all for a given (location, date) - the more consequential
    kind, since it can silently leave a field blank downstream (this is
    exactly what caused a gap in Best's Melbourne trend line: its top
    candidate for several targets had zero forecast rows for one date). An
    "unexpected null" is a populated row's field being null despite that
    (entity, field) pair having reported a real value at least once
    elsewhere in its own history - so a source that structurally never
    covers a field (BOM has no wind/humidity/cloud/pressure numerics;
    wttr.in has no precipitation total) doesn't inflate this number; only
    genuine anomalies do.

    Returns {"__ALL__": {...}, "<location name>": {...}, ...}, each value
    shaped {"missing_pct": float, "null_pct": float, "entities": [{"name":
    str, "missing_pct": float, "null_pct": float}, ...]} - ready to embed
    directly as the report's location-reactive data-quality section.
    """
    if not location_names:
        return {"__ALL__": {"missing_pct": 0.0, "null_pct": 0.0, "entities": []}}
    placeholders = ",".join("?" for _ in location_names)

    with db.connect(db_path) as conn:
        raw = pd.read_sql_query(
            f"SELECT source, location_name, forecast_date, collected_at, collection_method, "
            f"{', '.join(_RAW_FIELDS)} FROM forecasts WHERE location_name IN ({placeholders})",
            conn,
            params=location_names,
        )
        meta_frames = {}
        for model, table in _META_TABLES.items():
            cols = ["location_name", "forecast_date", "generated_at", *_META_FIELDS[model]]
            meta_frames[model] = pd.read_sql_query(
                f"SELECT {', '.join(cols)} FROM {table} WHERE location_name IN ({placeholders})",
                conn,
                params=location_names,
            )

    if raw.empty and all(f.empty for f in meta_frames.values()):
        return {"__ALL__": {"missing_pct": 0.0, "null_pct": 0.0, "entities": []}}

    all_dates = [pd.to_datetime(raw["forecast_date"])] if not raw.empty else []
    all_dates += [pd.to_datetime(f["forecast_date"]) for f in meta_frames.values() if not f.empty]
    anchor = max(s.max() for s in all_dates)
    window_start = anchor - pd.Timedelta(days=window_days - 1)

    # (entity_name, location_name) -> {"missing": n, "missing_total": n, "null": n, "null_total": n}
    entity_metrics: dict[tuple[str, str], dict[str, int]] = {}

    if not raw.empty:
        raw = raw.copy()
        raw["forecast_date"] = pd.to_datetime(raw["forecast_date"])
        raw_capability = {source: {f for f in _RAW_FIELDS if g[f].notna().any()} for source, g in raw.groupby("source")}
        first_seen = raw.groupby(["source", "location_name"])["forecast_date"].min()

        raw_window = raw[raw["forecast_date"] >= window_start].copy()
        # Live over backfill, then newest collected_at - same precedence load_modelling_table uses everywhere else.
        raw_window["_priority"] = (raw_window["collection_method"] != "live").astype(int)
        raw_window = raw_window.sort_values(["_priority", "collected_at"], ascending=[True, False])
        raw_latest = raw_window.drop_duplicates(subset=["source", "location_name", "forecast_date"], keep="first")

        for source in raw["source"].unique():
            capability = raw_capability.get(source, set())
            source_rows = raw_latest[raw_latest["source"] == source]
            for loc in location_names:
                fs = first_seen.get((source, loc))
                loc_rows = source_rows[source_rows["location_name"] == loc]
                existing_by_date = {r["forecast_date"]: r for _, r in loc_rows.iterrows()}
                entity_metrics[(source, loc)] = _entity_metrics_for_location(
                    existing_by_date, capability, _RAW_FIELDS, fs, window_start, anchor
                )

    for model, table_df in meta_frames.items():
        if table_df.empty:
            continue
        df = table_df.copy()
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        fields = _META_FIELDS[model]
        capability = {f for f in fields if df[f].notna().any()}
        first_seen_meta = df.groupby("location_name")["forecast_date"].min()

        window_df = df[df["forecast_date"] >= window_start].sort_values("generated_at", ascending=False)
        latest = window_df.drop_duplicates(subset=["location_name", "forecast_date"], keep="first")

        for loc in location_names:
            fs = first_seen_meta.get(loc)
            loc_rows = latest[latest["location_name"] == loc]
            existing_by_date = {r["forecast_date"]: r for _, r in loc_rows.iterrows()}
            entity_metrics[(model, loc)] = _entity_metrics_for_location(
                existing_by_date, capability, fields, fs, window_start, anchor
            )

    def _summarize(pairs: list[tuple[str, str]]) -> tuple[int, int, int, int]:
        m_n = sum(entity_metrics[p]["missing"] for p in pairs)
        m_t = sum(entity_metrics[p]["missing_total"] for p in pairs)
        n_n = sum(entity_metrics[p]["null"] for p in pairs)
        n_t = sum(entity_metrics[p]["null_total"] for p in pairs)
        return m_n, m_t, n_n, n_t

    entity_names = sorted({name for name, _loc in entity_metrics})
    result: dict[str, Any] = {}

    for loc in location_names:
        pairs = [(name, loc) for name in entity_names if (name, loc) in entity_metrics]
        m_n, m_t, n_n, n_t = _summarize(pairs)
        result[loc] = {
            "missing_pct": _pct(m_n, m_t),
            "null_pct": _pct(n_n, n_t),
            "entities": [
                {
                    "name": name,
                    "missing_pct": _pct(entity_metrics[(name, loc)]["missing"], entity_metrics[(name, loc)]["missing_total"]),
                    "null_pct": _pct(entity_metrics[(name, loc)]["null"], entity_metrics[(name, loc)]["null_total"]),
                }
                for name in entity_names
                if (name, loc) in entity_metrics
            ],
        }

    all_pairs = list(entity_metrics.keys())
    m_n, m_t, n_n, n_t = _summarize(all_pairs)
    pooled_entities = []
    for name in entity_names:
        name_pairs = [(name, loc) for loc in location_names if (name, loc) in entity_metrics]
        pm_n, pm_t, pn_n, pn_t = _summarize(name_pairs)
        pooled_entities.append({"name": name, "missing_pct": _pct(pm_n, pm_t), "null_pct": _pct(pn_n, pn_t)})

    result["__ALL__"] = {"missing_pct": _pct(m_n, m_t), "null_pct": _pct(n_n, n_t), "entities": pooled_entities}
    return result
