from __future__ import annotations

from datetime import date, datetime

import requests

from weather_ensemble.config import Location, TIMEOUT_SECONDS
from weather_ensemble.models import ForecastRecord


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def fetch_forecast(location: Location) -> ForecastRecord:
    """Fetch tomorrow's forecast from wttr.in."""
    url = f"https://wttr.in/{location.lat},{location.lon}"
    response = requests.get(url, params={"format": "j1"}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    try:
        day = payload["weather"][1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected wttr.in response structure") from exc

    hourly = day.get("hourly", [])
    rain_probability = max((_to_float(h.get("chanceofrain")) or 0 for h in hourly), default=None)
    uv_index = max((_to_float(h.get("uvIndex")) or 0 for h in hourly), default=None)
    wind_speed = max((_to_float(h.get("windspeedKmph")) or 0 for h in hourly), default=None)

    return ForecastRecord(
        source="wttr_in",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=date.fromisoformat(day["date"]),
        collected_at=datetime.now(),
        max_temp=_to_float(day.get("maxtempC")),
        min_temp=_to_float(day.get("mintempC")),
        rain_probability=rain_probability,
        uv_index=uv_index,
        wind_speed=wind_speed,
        raw_json=payload,
    )
