from __future__ import annotations

import os
from datetime import date, datetime

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
    """Fetch tomorrow's forecast from WeatherAPI.com.

    Requires WEATHERAPI_KEY in your .env. This collector is live-forecast only;
    WeatherAPI's normal history endpoint returns observations, not archived
    forecasts, so historical backfill is still handled by Open-Meteo.
    """
    api_key = os.getenv("WEATHERAPI_KEY")
    if not api_key:
        raise RuntimeError("WEATHERAPI_KEY is not set. Add it to .env to enable WeatherAPI.")

    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": f"{location.lat},{location.lon}",
        "days": 2,
        "aqi": "no",
        "alerts": "no",
    }
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    try:
        day = payload["forecast"]["forecastday"][1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected WeatherAPI response structure") from exc

    day_info = day.get("day", {})
    hours = day.get("hour", [])

    precip_sum = _to_float(day_info.get("totalprecip_mm"))
    weather_code = None
    condition = day_info.get("condition")
    if isinstance(condition, dict):
        weather_code = _to_float(condition.get("code"))

    return ForecastRecord(
        source="weatherapi",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=date.fromisoformat(day["date"]),
        collected_at=datetime.now(),
        max_temp=_to_float(day_info.get("maxtemp_c")),
        min_temp=_to_float(day_info.get("mintemp_c")),
        rain_probability=_to_float(day_info.get("daily_chance_of_rain")),
        precipitation_sum=precip_sum,
        uv_index=_to_float(day_info.get("uv")),
        wind_speed=_to_float(day_info.get("maxwind_kph")),
        wind_gusts=_max(hours, "gust_kph"),
        cloud_cover=_mean(hours, "cloud"),
        humidity=_to_float(day_info.get("avghumidity")) or _mean(hours, "humidity"),
        pressure_msl=_mean(hours, "pressure_mb"),
        weather_code=weather_code,
        raw_json=payload,
    )
