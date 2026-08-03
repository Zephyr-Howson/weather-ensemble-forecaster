from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from weather_ensemble import db
from weather_ensemble.config import PERIODS, Location, get_periods_db_path, local_today

# Haiku, not Opus/Sonnet: this writes a 2-3 sentence summary from ~10 numbers,
# once a night per location - the cheapest tier is the right fit for a
# budget-conscious nightly job, not a cost/quality tradeoff worth spending
# more on. See narrative.md discussion - user confirmed Haiku explicitly.
MODEL = "claude-haiku-4-5"
MAX_OUTPUT_TOKENS = 200

SYSTEM_PROMPT = (
    """
    You are a weather reporter. You will write a single brief weather summary from the
    forecast data given and some general definitions of weather metrics below. Cover:
    - Max and min temperature (assume min in the morning and max in the afternoon).
    - Whether it will rain, and if so how much it will rain and roughly when in the day.
    - Whether it will be sunny or cloudy.
    - Whether it will be windy (use wind speeds to define the level of windiness, only 
    quote wind gusts if extreme). 
    
    The target audience is a regular person who wants a brief summary of that day's 
    weather. Refer to the day by the name given in the forecast data (e.g. Monday).  
    Never say "tomorrow" or "today", since this may be read on a  different day than it 
    was written. Do not elaborate about what to wear or securing loose items. Be 
    consistent when describing the weather, and do not contradict yourself. 
    
    Plain prose only - no headings, bullet points, or markdown formatting. Do not mention 
    data sources, models, probabilities as percentages, or units you were not given.
    Round to the nearest whole number when referencing units.

    Temperature:
    Below 0° - freezing
    0-10°C - cold
    11-15°C - chilly / brisk
    16-19°C - cool
    20-22°C - mild
    23-27°C - warm
    28-34°C - hot
    35°+ - scorching / boiling

    Rain:
    0 mm - dry - No rain expected.
    0.1-4 mm - drizzles - Brief, light rain.
    0.1-2 mm - showers - Brief, passing rain.
    5-19 mm - wet - Continuous light rain or heavy bursts. You need an umbrella.
    20-49 mm - heavy rain - Heavy, persistent rain all day.
    50+ mm - severe rain - Torrential downpours. Flash flooding is likely. Warnings are issued.
    
    Cloud cover:
    0%-10% cloud cover - sunny
    11%-30% cloud cover - mostly sunny
    31%-60% cloud cover - partly cloudy
    61%-90% cloud cover - mostly cloudy
    91-100% cloud cover - overcast
    
    Wind:
    Below 11 km/h wind speed - calm
    12-19 km/h wind speed - pleasant
    20-28 km/h wind speed - breezy
    29-38 km/h wind speed - windy
    39-61 km/h wind speed - very windy
    62+ km/h wind speed - severe / dangerous
    """
)


def _fetch_best_prediction(db_path: Path, location: Location, target_date: date) -> dict[str, Any] | None:
    with db.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM best_predictions WHERE location_name = ? AND forecast_date = ? "
            "ORDER BY generated_at DESC LIMIT 1",
            (location.name, target_date.isoformat()),
        ).fetchone()
    return dict(row) if row is not None else None


def _fetch_best_periods(db_path: Path, location: Location, target_date: date) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    with db.connect_periods(get_periods_db_path(db_path)) as conn:
        for period in PERIODS:
            row = conn.execute(
                "SELECT precipitation_sum FROM best_predictions_periods "
                "WHERE location_name = ? AND forecast_date = ? AND period = ? "
                "ORDER BY generated_at DESC LIMIT 1",
                (location.name, target_date.isoformat(), period),
            ).fetchone()
            values[period] = row["precipitation_sum"] if row is not None else None
    return values


def _build_user_prompt(
    location: Location, target_date: date, prediction: dict[str, Any], period_precip: dict[str, float | None]
) -> str:
    lines = [
        f"Location: {location.name}",
        f"Day: {target_date.strftime('%A')}",
        f"Date: {target_date.isoformat()}",
    ]

    max_temp, min_temp = prediction.get("max_temp"), prediction.get("min_temp")
    if max_temp is not None and min_temp is not None:
        lines.append(f"Max/min temperature: {max_temp:.1f}C / {min_temp:.1f}C")

    rain_prob = prediction.get("did_rain_probability")
    if rain_prob is not None:
        lines.append(f"Chance of rain: {rain_prob * 100:.0f}%")

    precip = prediction.get("precipitation_sum")
    if precip is not None:
        lines.append(f"Total precipitation: {precip:.1f}mm")

    period_bits = [f"{period} {value:.1f}mm" for period, value in period_precip.items() if value is not None]
    if period_bits:
        lines.append("Precipitation by period: " + ", ".join(period_bits))

    cloud_cover = prediction.get("cloud_cover")
    if cloud_cover is not None:
        lines.append(f"Cloud cover: {cloud_cover:.0f}%")

    wind_speed, wind_gusts = prediction.get("wind_speed"), prediction.get("wind_gusts")
    if wind_speed is not None:
        gust_bit = f" (gusts {wind_gusts:.0f} km/h)" if wind_gusts is not None else ""
        lines.append(f"Wind: {wind_speed:.0f} km/h{gust_bit}")

    return "\n".join(lines)


def _call_llm(prompt: str) -> str | None:
    """The one call this whole feature makes. Any failure at all - no API
    key configured, the package missing, a network error, a non-2xx
    response, a malformed/empty reply - returns None rather than raising, so
    the caller can silently skip this location's narrative for tonight
    rather than crash the run or show a broken segment on the report. The
    SDK already retries transient failures (429/5xx/connection errors) a
    couple of times on its own before this catch is ever reached.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception:  # noqa: BLE001 - deliberately catch-all, see docstring
        return None

    text = next((block.text for block in response.content if block.type == "text"), None)
    return text.strip() if text else None


def generate_and_store_best_narrative(
    db_path: Path, location: Location, target_date: date | None = None
) -> dict[str, Any]:
    """Generate and persist a brief natural-language narrative for Best's
    target_date prediction (default: tomorrow) via a single Claude API call.

    Best never predicts anything itself - see best.py - so this reads
    whatever Best already picked (the same values report.py's recent-
    forecasts table shows) rather than recomputing anything. Any failure
    anywhere in this pipeline (no Best prediction yet, the LLM call failing
    for any reason) returns a "skipped" result rather than raising - callers
    should treat this exactly like the existing _guarded per-location steps
    in cli.py: log and move on, never abort the run.
    """
    if target_date is None:
        target_date = local_today(location) + timedelta(days=1)

    prediction = _fetch_best_prediction(db_path, location, target_date)
    if prediction is None:
        return {"skipped": "no_best_prediction", "forecast_date": target_date.isoformat()}

    period_precip = _fetch_best_periods(db_path, location, target_date)
    prompt = _build_user_prompt(location, target_date, prediction, period_precip)

    narrative = _call_llm(prompt)
    if narrative is None:
        return {"skipped": "llm_unavailable", "forecast_date": target_date.isoformat()}

    generated_at = datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")
    with db.connect(db_path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO best_narratives "
            "(location_name, lat, lon, forecast_date, generated_at, model, narrative) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (location.name, location.lat, location.lon, target_date.isoformat(), generated_at, MODEL, narrative),
        )
        conn.commit()

    return {"forecast_date": target_date.isoformat(), "model": MODEL, "narrative": narrative}
