"""
Weather Ensemble Forecaster - Phase 1
======================================
Collects forecasts from multiple free weather APIs, scores historical
accuracy per source, and blends them using inverse-error weighting.

Variables tracked: max_temp, min_temp, rain_probability, uv_index, wind_speed

Sources used (all free, no API key required for basic use):
  - Open-Meteo       (https://open-meteo.com)
  - Open-Meteo WMO   (second model: ECMWF via same API)
  - wttr.in          (https://wttr.in)

Usage:
    python weather_ensemble.py --lat -37.8136 --lon 144.9631 --collect
    python weather_ensemble.py --lat -37.8136 --lon 144.9631 --record-actuals
    python weather_ensemble.py --lat -37.8136 --lon 144.9631 --forecast
    python weather_ensemble.py --lat -37.8136 --lon 144.9631 --all
    python weather_ensemble.py --lat -37.8136 --lon 144.9631 --report

Typical daily workflow (run once each morning):
    python weather_ensemble.py --lat LAT --lon LON --all
"""

import argparse
import json
import math
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import requests

# ── Configuration ────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "forecasts.db"

# Number of days of history used to compute weights (tunable)
ROLLING_WINDOW_DAYS = 30

# Variables we track and their human-readable labels
VARIABLES = {
    "max_temp":         "Max temperature (°C)",
    "min_temp":         "Min temperature (°C)",
    "rain_probability": "Rain probability (%)",
    "uv_index":         "UV index",
    "wind_speed":       "Wind speed (km/h)",
}

# Weather condition categories — used to split accuracy scores
# so a source that's great on sunny days but poor on rainy days
# gets appropriate weighting per condition.
CONDITION_CATEGORIES = {
    "clear":  "Clear / sunny (rain prob < 20%)",
    "cloudy": "Cloudy / overcast (rain prob 20–60%)",
    "rainy":  "Rainy (rain prob > 60%)",
}

# Request timeout in seconds
TIMEOUT = 15


# ── Database setup ────────────────────────────────────────────────────────────

def init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't already exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            forecast_date   TEXT    NOT NULL,   -- ISO date being forecast for
            collected_at    TEXT    NOT NULL,   -- ISO datetime when collected
            source          TEXT    NOT NULL,   -- source identifier
            lat             REAL    NOT NULL,
            lon             REAL    NOT NULL,
            max_temp        REAL,
            min_temp        REAL,
            rain_probability REAL,
            uv_index        REAL,
            wind_speed      REAL,
            raw_json        TEXT                -- full API response for debugging
        );

        CREATE TABLE IF NOT EXISTS actuals (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            actual_date   TEXT NOT NULL,        -- ISO date of the observation
            lat           REAL NOT NULL,
            lon           REAL NOT NULL,
            max_temp      REAL,
            min_temp      REAL,
            rain_probability REAL,              -- estimated from precipitation sum
            uv_index      REAL,
            wind_speed    REAL,
            raw_json      TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS uq_actuals
            ON actuals(actual_date, lat, lon);

        CREATE TABLE IF NOT EXISTS accuracy_scores (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scored_at       TEXT NOT NULL,
            source          TEXT NOT NULL,
            variable        TEXT NOT NULL,
            condition_type  TEXT NOT NULL,
            window_days     INTEGER NOT NULL,
            lat             REAL NOT NULL,
            lon             REAL NOT NULL,
            mae             REAL,               -- mean absolute error
            sample_count    INTEGER
        );
    """)
    conn.commit()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


# ── API collectors ────────────────────────────────────────────────────────────

def fetch_open_meteo(lat: float, lon: float, model: str = "best_match") -> dict:
    """
    Fetch tomorrow's forecast from Open-Meteo.
    model can be 'best_match', 'bom_access_global', 'ecmwf_ifs025', etc.
    Note: not all models support all variables (e.g. some lack UV or precip
    probability). We request all and gracefully return None for missing fields.
    https://open-meteo.com/en/docs
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "uv_index_max",
            "wind_speed_10m_max",
        ]),
        "timezone": "auto",
        "forecast_days": 3,
        "models": model,
    }
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # Index 1 = tomorrow (index 0 = today)
    daily = data["daily"]
    tomorrow_idx = 1

    def safe_get(key):
        """Return value at tomorrow_idx, or None if the field is missing or all-None."""
        vals = daily.get(key)
        if not vals:
            return None
        val = vals[tomorrow_idx] if tomorrow_idx < len(vals) else None
        return val  # may legitimately be None if the model doesn't provide it

    return {
        "max_temp":         safe_get("temperature_2m_max"),
        "min_temp":         safe_get("temperature_2m_min"),
        "rain_probability": safe_get("precipitation_probability_max"),
        "uv_index":         safe_get("uv_index_max"),
        "wind_speed":       safe_get("wind_speed_10m_max"),
        "_raw":             data,
    }


def fetch_wttr(lat: float, lon: float) -> dict:
    """
    Fetch tomorrow's forecast from wttr.in (uses Dark Sky / Met.no backend).
    https://wttr.in/:help
    """
    url = f"https://wttr.in/{lat},{lon}"
    params = {"format": "j1"}
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    # weather[0] = today, weather[1] = tomorrow
    try:
        day = data["weather"][1]
        hourly = day.get("hourly", [])

        # UV: take max across hourly slots
        uv_vals = [float(h.get("uvIndex", 0)) for h in hourly if h.get("uvIndex") is not None]
        uv_max = max(uv_vals) if uv_vals else None

        # Rain probability: max of chance_of_rain across hourly
        rain_vals = [float(h.get("chanceofrain", 0)) for h in hourly]
        rain_max = max(rain_vals) if rain_vals else None

        # Wind: max windspeedKmph
        wind_vals = [float(h.get("windspeedKmph", 0)) for h in hourly]
        wind_max = max(wind_vals) if wind_vals else None

        return {
            "max_temp":         float(day["maxtempC"]),
            "min_temp":         float(day["mintempC"]),
            "rain_probability": rain_max,
            "uv_index":         uv_max,
            "wind_speed":       wind_max,
            "_raw":             data,
        }
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Unexpected wttr.in response structure: {e}") from e


# Registry of all sources: name → callable(lat, lon) → dict
SOURCES = {
    "open_meteo_best":  lambda lat, lon: fetch_open_meteo(lat, lon, model="best_match"),
    "open_meteo_bom":   lambda lat, lon: fetch_open_meteo(lat, lon, model="bom_access_global"),
    "wttr_in":          fetch_wttr,
}


# ── Step 1: Collect forecasts ─────────────────────────────────────────────────

def collect_forecasts(lat: float, lon: float) -> None:
    """Fetch tomorrow's forecasts from all sources and store in DB."""
    from datetime import datetime
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    collected_at = datetime.now().isoformat(timespec="seconds")

    conn = get_conn()
    print(f"\n📡  Collecting forecasts for {lat}, {lon} (target date: {tomorrow})")

    for source_name, fetcher in SOURCES.items():
        print(f"   → {source_name} ...", end=" ", flush=True)
        try:
            result = fetcher(lat, lon)
            raw_json = json.dumps(result.pop("_raw", {}))
            conn.execute("""
                INSERT INTO forecasts
                    (forecast_date, collected_at, source, lat, lon,
                     max_temp, min_temp, rain_probability, uv_index, wind_speed, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tomorrow, collected_at, source_name, lat, lon,
                result.get("max_temp"),
                result.get("min_temp"),
                result.get("rain_probability"),
                result.get("uv_index"),
                result.get("wind_speed"),
                raw_json,
            ))
            print(f"✓  max={result.get('max_temp')}°C  min={result.get('min_temp')}°C  "
                  f"rain={result.get('rain_probability')}%  "
                  f"uv={result.get('uv_index')}  wind={result.get('wind_speed')}km/h")
        except Exception as e:
            print(f"✗  ERROR: {e}")

    conn.commit()
    conn.close()
    print("   Collection complete.\n")


# ── Step 2: Record actuals ────────────────────────────────────────────────────

def fetch_actuals_open_meteo(lat: float, lon: float, target_date: date) -> dict:
    """
    Fetch observed actuals for a past date using Open-Meteo's historical API.
    For rain probability we approximate from precipitation sum > 1mm → 100%, else 0%.
    In practice you can refine this with more granular thresholds.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    date_str = target_date.isoformat()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "uv_index_max",
            "wind_speed_10m_max",
        ]),
        "timezone": "auto",
    }
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    daily = data["daily"]

    precip = daily.get("precipitation_sum", [None])[0]
    # Approximate rain probability from actual precipitation
    if precip is None:
        rain_prob = None
    elif precip >= 5.0:
        rain_prob = 90.0
    elif precip >= 1.0:
        rain_prob = 70.0
    elif precip > 0.0:
        rain_prob = 30.0
    else:
        rain_prob = 5.0

    return {
        "max_temp":         daily.get("temperature_2m_max", [None])[0],
        "min_temp":         daily.get("temperature_2m_min", [None])[0],
        "rain_probability": rain_prob,
        "uv_index":         daily.get("uv_index_max", [None])[0],
        "wind_speed":       daily.get("wind_speed_10m_max", [None])[0],
        "_raw":             data,
    }


def record_actuals(lat: float, lon: float, target_date: date | None = None) -> None:
    """
    Record observed actuals for yesterday (default) or a specified date.
    Safe to re-run — uses INSERT OR REPLACE.
    """
    from datetime import datetime
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    print(f"\n📋  Recording actuals for {lat}, {lon} on {target_date.isoformat()} ...", end=" ", flush=True)
    try:
        result = fetch_actuals_open_meteo(lat, lon, target_date)
        raw_json = json.dumps(result.pop("_raw", {}))
        conn = get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO actuals
                (actual_date, lat, lon, max_temp, min_temp,
                 rain_probability, uv_index, wind_speed, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            target_date.isoformat(), lat, lon,
            result.get("max_temp"),
            result.get("min_temp"),
            result.get("rain_probability"),
            result.get("uv_index"),
            result.get("wind_speed"),
            raw_json,
        ))
        conn.commit()
        conn.close()
        print(f"✓  max={result.get('max_temp')}°C  min={result.get('min_temp')}°C  "
              f"rain≈{result.get('rain_probability')}%  "
              f"uv={result.get('uv_index')}  wind={result.get('wind_speed')}km/h")
    except Exception as e:
        print(f"✗  ERROR: {e}")
    print()


# ── Step 3: Score accuracy ────────────────────────────────────────────────────

def classify_condition(rain_probability: float | None) -> str:
    """Classify a day's weather condition from its rain probability."""
    if rain_probability is None:
        return "clear"
    if rain_probability < 20:
        return "clear"
    if rain_probability <= 60:
        return "cloudy"
    return "rainy"


def compute_scores(lat: float, lon: float, window_days: int = ROLLING_WINDOW_DAYS) -> dict:
    """
    Compute MAE per (source, variable, condition_type) over the rolling window.
    Returns a nested dict: scores[source][variable][condition] = {"mae": float, "n": int}
    """
    conn = get_conn()
    cutoff = (date.today() - timedelta(days=window_days)).isoformat()

    # Pull all forecasts within window that have a matching actual
    rows = conn.execute("""
        SELECT f.source, f.forecast_date,
               f.max_temp, f.min_temp, f.rain_probability, f.uv_index, f.wind_speed,
               a.max_temp AS a_max_temp, a.min_temp AS a_min_temp,
               a.rain_probability AS a_rain_prob, a.uv_index AS a_uv,
               a.wind_speed AS a_wind,
               a.rain_probability AS actual_rain_prob
        FROM forecasts f
        JOIN actuals a
            ON f.forecast_date = a.actual_date
            AND ABS(f.lat - a.lat) < 0.01
            AND ABS(f.lon - a.lon) < 0.01
        WHERE f.forecast_date >= ?
          AND ABS(f.lat - ?) < 0.01
          AND ABS(f.lon - ?) < 0.01
        ORDER BY f.source, f.forecast_date
    """, (cutoff, lat, lon)).fetchall()
    conn.close()

    if not rows:
        return {}

    # Accumulate absolute errors per (source, variable, condition)
    # Structure: errors[source][variable][condition] = [abs_error, ...]
    errors: dict = {}

    for row in rows:
        source = row["source"]
        condition = classify_condition(row["actual_rain_prob"])

        if source not in errors:
            errors[source] = {v: {"clear": [], "cloudy": [], "rainy": []} for v in VARIABLES}

        pairs = [
            ("max_temp",         row["max_temp"],         row["a_max_temp"]),
            ("min_temp",         row["min_temp"],         row["a_min_temp"]),
            ("rain_probability", row["rain_probability"], row["a_rain_prob"]),
            ("uv_index",         row["uv_index"],         row["a_uv"]),
            ("wind_speed",       row["wind_speed"],       row["a_wind"]),
        ]
        for var, pred, actual in pairs:
            if pred is not None and actual is not None:
                errors[source][var][condition].append(abs(pred - actual))

    # Convert to MAE
    scores: dict = {}
    for source, var_dict in errors.items():
        scores[source] = {}
        for var, cond_dict in var_dict.items():
            scores[source][var] = {}
            for cond, errs in cond_dict.items():
                if errs:
                    scores[source][var][cond] = {
                        "mae": sum(errs) / len(errs),
                        "n":   len(errs),
                    }
                else:
                    scores[source][var][cond] = {"mae": None, "n": 0}

    return scores


# ── Step 4: Blend forecasts ───────────────────────────────────────────────────

def blend_forecasts(lat: float, lon: float, window_days: int = ROLLING_WINDOW_DAYS) -> dict:
    """
    Produce a weighted-ensemble forecast for tomorrow.

    Weights are based on inverse MAE per (source, variable, condition_type).
    When insufficient history exists for a condition, equal weights are used.

    Returns a dict with blended values plus per-variable metadata.
    """
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    conn = get_conn()
    rows = conn.execute("""
        SELECT source, max_temp, min_temp, rain_probability, uv_index, wind_speed
        FROM forecasts
        WHERE forecast_date = ?
          AND ABS(lat - ?) < 0.01
          AND ABS(lon - ?) < 0.01
        ORDER BY collected_at DESC
    """, (tomorrow, lat, lon)).fetchall()
    conn.close()

    if not rows:
        return {"error": "No forecasts found for tomorrow. Run --collect first."}

    # Deduplicate: keep most-recent row per source
    seen = {}
    for row in rows:
        if row["source"] not in seen:
            seen[row["source"]] = dict(row)
    sources_today = list(seen.values())

    # Get accuracy scores
    scores = compute_scores(lat, lon, window_days)

    # Estimate tomorrow's condition from average rain probability to choose weight bucket
    rain_preds = [s["rain_probability"] for s in sources_today if s["rain_probability"] is not None]
    avg_rain = sum(rain_preds) / len(rain_preds) if rain_preds else None
    condition = classify_condition(avg_rain)

    blended = {}
    metadata = {}

    for var in VARIABLES:
        predictions = [(s["source"], s[var]) for s in sources_today if s[var] is not None]
        if not predictions:
            blended[var] = None
            continue

        # Build per-source weights from inverse MAE for this variable + condition
        weights = []
        weight_details = []
        for src, pred in predictions:
            mae = None
            n = 0
            if scores and src in scores and var in scores[src]:
                entry = scores[src][var].get(condition, {"mae": None, "n": 0})
                mae = entry["mae"]
                n = entry["n"]

            if mae is not None and mae > 0 and n >= 3:
                w = 1.0 / mae
            else:
                # Fall back to equal weighting when insufficient data
                w = 1.0

            weights.append(w)
            weight_details.append({
                "source": src,
                "prediction": pred,
                "weight": w,
                "mae": mae,
                "sample_count": n,
            })

        total_w = sum(weights)
        blended_val = sum(p * w for (_, p), w in zip(predictions, weights)) / total_w

        blended[var] = round(blended_val, 2)
        metadata[var] = {
            "condition_used": condition,
            "sources": [
                {**d, "weight_pct": round(d["weight"] / total_w * 100, 1)}
                for d in weight_details
            ],
        }

    return {"date": tomorrow, "blended": blended, "metadata": metadata, "condition": condition}


# ── Step 5: Persist scores ────────────────────────────────────────────────────

def save_scores(lat: float, lon: float, window_days: int = ROLLING_WINDOW_DAYS) -> None:
    """Compute and save current accuracy scores to DB (for tracking over time)."""
    from datetime import datetime
    scores = compute_scores(lat, lon, window_days)
    if not scores:
        print("   No historical data to score yet.\n")
        return

    conn = get_conn()
    now = datetime.now().isoformat(timespec="seconds")
    for source, var_dict in scores.items():
        for var, cond_dict in var_dict.items():
            for cond, entry in cond_dict.items():
                if entry["n"] > 0:
                    conn.execute("""
                        INSERT INTO accuracy_scores
                            (scored_at, source, variable, condition_type,
                             window_days, lat, lon, mae, sample_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (now, source, var, cond, window_days, lat, lon,
                          entry["mae"], entry["n"]))
    conn.commit()
    conn.close()


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_forecast(result: dict) -> None:
    """Pretty-print the blended forecast."""
    if "error" in result:
        print(f"\n⚠️   {result['error']}\n")
        return

    print(f"\n{'═' * 58}")
    print(f"  🌤  Ensemble Forecast — {result['date']}")
    print(f"  Condition category: {result['condition']}")
    print(f"{'═' * 58}")

    blended = result["blended"]
    meta    = result.get("metadata", {})

    var_display = [
        ("max_temp",         "Max temperature",   "°C"),
        ("min_temp",         "Min temperature",   "°C"),
        ("rain_probability", "Rain probability",  "%"),
        ("uv_index",         "UV index",          ""),
        ("wind_speed",       "Wind speed",        " km/h"),
    ]

    for key, label, unit in var_display:
        val = blended.get(key)
        val_str = f"{val}{unit}" if val is not None else "N/A"
        print(f"  {label:<22} {val_str}")

    print(f"\n{'─' * 58}")
    print("  Source weights used:")
    for key, label, _ in var_display:
        if key not in meta:
            continue
        src_info = meta[key]["sources"]
        detail = "  ".join(
            f"{s['source']} {s['weight_pct']}%"
            + (f" (MAE={s['mae']:.2f}, n={s['sample_count']})" if s['mae'] else " (equal/fallback)")
            for s in src_info
        )
        print(f"  {label:<22} {detail}")

    print(f"{'═' * 58}\n")


def print_accuracy_report(lat: float, lon: float, window_days: int = ROLLING_WINDOW_DAYS) -> None:
    """Print a summary of each source's accuracy per variable and condition."""
    scores = compute_scores(lat, lon, window_days)
    if not scores:
        print("\n⚠️   No scored history yet. Collect forecasts for a few days, then run --record-actuals.\n")
        return

    print(f"\n{'═' * 70}")
    print(f"  📊  Accuracy Report — {window_days}-day rolling window")
    print(f"{'═' * 70}")

    for source in sorted(scores):
        print(f"\n  Source: {source}")
        print(f"  {'Variable':<24} {'Clear MAE':>12} {'Cloudy MAE':>12} {'Rainy MAE':>12}")
        print(f"  {'─' * 62}")
        for var, label in VARIABLES.items():
            cond_data = scores[source].get(var, {})
            def fmt(c):
                d = cond_data.get(c, {})
                if d.get("n", 0) == 0:
                    return "      —"
                return f"{d['mae']:>7.2f} (n={d['n']})"
            print(f"  {label:<24} {fmt('clear'):>12} {fmt('cloudy'):>12} {fmt('rainy'):>12}")

    print(f"\n{'═' * 70}\n")


def print_db_summary(lat: float, lon: float) -> None:
    """Print a quick summary of what's in the database."""
    conn = get_conn()
    n_forecasts = conn.execute(
        "SELECT COUNT(*) FROM forecasts WHERE ABS(lat-?) < 0.01 AND ABS(lon-?) < 0.01",
        (lat, lon)).fetchone()[0]
    n_actuals = conn.execute(
        "SELECT COUNT(*) FROM actuals WHERE ABS(lat-?) < 0.01 AND ABS(lon-?) < 0.01",
        (lat, lon)).fetchone()[0]
    dates = conn.execute(
        "SELECT MIN(forecast_date), MAX(forecast_date) FROM forecasts WHERE ABS(lat-?) < 0.01 AND ABS(lon-?) < 0.01",
        (lat, lon)).fetchone()
    conn.close()

    print(f"\n  DB: {n_forecasts} forecast rows, {n_actuals} actual rows  "
          f"| forecast range: {dates[0]} → {dates[1]}\n")


# ── Backfill utilities ────────────────────────────────────────────────────────

def backfill_actuals(lat: float, lon: float, days_back: int = 30) -> None:
    """
    Backfill actuals for the past N days — useful when first setting up
    so you have historical data to score against from day one.
    """
    print(f"\n🔄  Backfilling actuals for the past {days_back} days...\n")
    for i in range(1, days_back + 1):
        target = date.today() - timedelta(days=i)
        record_actuals(lat, lon, target_date=target)


def backfill_forecast_source(
    lat: float, lon: float, days_back: int, source_name: str, model: str
) -> int:
    """
    Reconstruct historical forecasts for one Open-Meteo model using the
    past_days parameter, which returns what the model predicted on each
    past date as its day-ahead forecast.
    Returns the number of rows inserted.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "uv_index_max",
            "wind_speed_10m_max",
        ]),
        "timezone": "auto",
        "past_days": days_back,
        "forecast_days": 1,
        "models": model,
    }

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"      ✗  {source_name}: API error — {e}")
        return 0

    daily = data["daily"]
    dates = daily.get("time", [])

    def col(key):
        return daily.get(key, [None] * len(dates))

    conn = get_conn()
    inserted = 0

    for i, d in enumerate(dates):
        if d >= date.today().isoformat():
            continue
        existing = conn.execute(
            "SELECT id FROM forecasts WHERE forecast_date=? AND source=? AND ABS(lat-?)<0.01 AND ABS(lon-?)<0.01",
            (d, source_name, lat, lon)
        ).fetchone()
        if existing:
            continue

        # past_days responses represent the day-ahead forecast issued ~06:00 local
        collected_at = f"{d}T06:00:00"

        def safe(lst, idx=i):
            return lst[idx] if idx < len(lst) else None

        conn.execute("""
            INSERT INTO forecasts
                (forecast_date, collected_at, source, lat, lon,
                 max_temp, min_temp, rain_probability, uv_index, wind_speed, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d, collected_at, source_name, lat, lon,
            safe(col("temperature_2m_max")),
            safe(col("temperature_2m_min")),
            safe(col("precipitation_probability_max")),
            safe(col("uv_index_max")),
            safe(col("wind_speed_10m_max")),
            "{}",
        ))
        inserted += 1

    conn.commit()
    conn.close()
    return inserted


def backfill_forecasts(lat: float, lon: float, days_back: int = 30) -> None:
    """
    Backfill historical forecasts for all Open-Meteo sources.
    wttr.in has no historical forecast endpoint so it is skipped —
    it will accumulate naturally from daily --collect runs.
    """
    om_sources = {
        "open_meteo_best": "best_match",
        "open_meteo_bom":  "bom_access_global",
    }

    print(f"\n🔄  Backfilling forecasts for the past {days_back} days...")
    print(f"    (wttr_in skipped — no historical forecast API)\n")

    for source_name, model in om_sources.items():
        print(f"   → {source_name} ({model}) ...", end=" ", flush=True)
        n = backfill_forecast_source(lat, lon, days_back, source_name, model)
        print(f"✓  {n} rows inserted")

    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Weather Ensemble Forecaster — Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  First-time setup (backfill 30 days of actuals + forecasts, then collect today):
    python weather_ensemble.py --lat -37.81 --lon 144.96 --backfill 30 --collect
    This gives you real MAE weights immediately rather than waiting weeks.

  Backfill forecasts only (if you already ran --backfill-actuals separately):
    python weather_ensemble.py --lat -37.81 --lon 144.96 --backfill-forecasts 30

  Daily run (collect today + record yesterday's actuals + show forecast):
    python weather_ensemble.py --lat -37.81 --lon 144.96 --all

  Just show the forecast:
    python weather_ensemble.py --lat -37.81 --lon 144.96 --forecast

  Accuracy report:
    python weather_ensemble.py --lat -37.81 --lon 144.96 --report

  Change rolling window to 14 days:
    python weather_ensemble.py --lat -37.81 --lon 144.96 --forecast --window 14
        """,
    )
    parser.add_argument("--lat",     type=float, required=True,  help="Latitude")
    parser.add_argument("--lon",     type=float, required=True,  help="Longitude")
    parser.add_argument("--window",  type=int,   default=ROLLING_WINDOW_DAYS,
                        help=f"Rolling window in days for accuracy scoring (default: {ROLLING_WINDOW_DAYS})")
    parser.add_argument("--collect",        action="store_true", help="Fetch and store today's forecasts")
    parser.add_argument("--record-actuals", action="store_true", help="Record yesterday's observed actuals")
    parser.add_argument("--forecast",       action="store_true", help="Show blended forecast for tomorrow")
    parser.add_argument("--report",         action="store_true", help="Print accuracy report")
    parser.add_argument("--all",            action="store_true", help="Run --collect + --record-actuals + --forecast")
    parser.add_argument("--backfill",          type=int, metavar="DAYS",
                        help="Backfill BOTH actuals and forecasts for the past N days (recommended first-time setup)")
    parser.add_argument("--backfill-actuals",  type=int, metavar="DAYS",
                        help="Backfill only observed actuals for the past N days")
    parser.add_argument("--backfill-forecasts",type=int, metavar="DAYS",
                        help="Backfill only historical forecasts (Open-Meteo sources) for the past N days")
    parser.add_argument("--db-summary",        action="store_true", help="Show DB row counts")

    args = parser.parse_args()

    if args.backfill:
        # Convenience: run both together so MAE scoring works immediately
        backfill_actuals(args.lat, args.lon, args.backfill)
        backfill_forecasts(args.lat, args.lon, args.backfill)

    if args.backfill_actuals:
        backfill_actuals(args.lat, args.lon, args.backfill_actuals)

    if args.backfill_forecasts:
        backfill_forecasts(args.lat, args.lon, args.backfill_forecasts)

    if args.collect or args.all:
        collect_forecasts(args.lat, args.lon)

    if args.record_actuals or args.all:
        record_actuals(args.lat, args.lon)

    if args.forecast or args.all:
        result = blend_forecasts(args.lat, args.lon, args.window)
        print_forecast(result)
        save_scores(args.lat, args.lon, args.window)

    if args.report:
        print_accuracy_report(args.lat, args.lon, args.window)

    if args.db_summary:
        print_db_summary(args.lat, args.lon)

    if not any([args.backfill, args.collect, args.record_actuals,
                args.forecast, args.report, args.all, args.db_summary]):
        parser.print_help()


if __name__ == "__main__":
    main()