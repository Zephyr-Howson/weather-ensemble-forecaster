from __future__ import annotations

import os
from datetime import date, datetime

from weather_ensemble.config import Location, TIMEOUT_SECONDS
from weather_ensemble.models import ForecastRecord
from weather_ensemble.retry import get_with_retry


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _kmh(value: object) -> float | None:
    """Weatherbit reports wind speed in m/s by default; convert to km/h."""
    ms = _to_float(value)
    return round(ms * 3.6, 3) if ms is not None else None


def fetch_forecast(location: Location) -> ForecastRecord:
    """Fetch tomorrow's forecast from Weatherbit's 16-day daily forecast endpoint.

    Requires WEATHERBIT_KEY in your .env. This collector is live-forecast only;
    historical backfill remains Open-Meteo's job.
    """
    api_key = os.getenv("WEATHERBIT_KEY")
    if not api_key:
        raise RuntimeError("WEATHERBIT_KEY is not set. Add it to .env to enable Weatherbit.")

    url = "https://api.weatherbit.io/v2.0/forecast/daily"
    params = {
        "lat": location.lat,
        "lon": location.lon,
        "key": api_key,
        "days": 2,
    }
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()

    try:
        day = payload["data"][1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected Weatherbit response structure") from exc

    weather = day.get("weather") or {}
    # Weatherbit's daily forecast exposes both surface pressure ("pres") and
    # sea-level pressure ("slp"); prefer slp to match the project's pressure_msl field.
    pressure = day.get("slp", day.get("pres"))

    return ForecastRecord(
        source="weatherbit",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=date.fromisoformat(day["valid_date"]),
        collected_at=datetime.now(),
        max_temp=_to_float(day.get("max_temp")),
        min_temp=_to_float(day.get("min_temp")),
        rain_probability=_to_float(day.get("pop")),
        precipitation_sum=_to_float(day.get("precip")),
        uv_index=_to_float(day.get("uv")),
        wind_speed=_kmh(day.get("wind_spd")),
        wind_gusts=_kmh(day.get("wind_gust_spd")),
        cloud_cover=_to_float(day.get("clouds")),
        humidity=_to_float(day.get("rh")),
        pressure_msl=_to_float(pressure),
        weather_code=_to_float(weather.get("code")),
        raw_json=payload,
    )
