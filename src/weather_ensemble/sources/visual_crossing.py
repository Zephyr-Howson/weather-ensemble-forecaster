from __future__ import annotations

import os
from datetime import date, datetime, timedelta

import requests

from weather_ensemble.config import Location, TIMEOUT_SECONDS
from weather_ensemble.models import ForecastRecord


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _max(items: list[dict], key: str) -> float | None:
    values = [_to_float(item.get(key)) for item in items]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _mean(items: list[dict], key: str) -> float | None:
    values = [_to_float(item.get(key)) for item in items]
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 3) if values else None


def fetch_forecast(location: Location) -> ForecastRecord:
    """Fetch tomorrow's forecast from Visual Crossing Timeline API.

    Requires VISUAL_CROSSING_KEY in your .env. This collector is live-forecast
    only in this project; historical forecast archive access varies by Visual
    Crossing account/endpoint, so Open-Meteo remains the default backfill source.
    """
    api_key = os.getenv("VISUAL_CROSSING_KEY")
    if not api_key:
        raise RuntimeError("VISUAL_CROSSING_KEY is not set. Add it to .env to enable Visual Crossing.")

    target = date.today() + timedelta(days=1)
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{location.lat},{location.lon}/{target.isoformat()}/{target.isoformat()}"
    )
    params = {
        "key": api_key,
        "unitGroup": "metric",
        "include": "days,hours",
        "contentType": "json",
    }
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    try:
        day = payload["days"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected Visual Crossing response structure") from exc

    hours = day.get("hours", [])

    return ForecastRecord(
        source="visual_crossing",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=date.fromisoformat(day["datetime"]),
        collected_at=datetime.now(),
        max_temp=_to_float(day.get("tempmax")),
        min_temp=_to_float(day.get("tempmin")),
        rain_probability=_to_float(day.get("precipprob")),
        precipitation_sum=_to_float(day.get("precip")),
        uv_index=_to_float(day.get("uvindex")),
        wind_speed=_to_float(day.get("windspeed")),
        wind_gusts=_to_float(day.get("windgust")) or _max(hours, "windgust"),
        cloud_cover=_to_float(day.get("cloudcover")) or _mean(hours, "cloudcover"),
        humidity=_to_float(day.get("humidity")) or _mean(hours, "humidity"),
        pressure_msl=_to_float(day.get("pressure")) or _mean(hours, "pressure"),
        weather_code=None,  # Visual Crossing uses textual conditions/icons rather than WMO codes.
        raw_json=payload,
    )
