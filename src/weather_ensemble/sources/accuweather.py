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


def _get_location_key(location: Location, api_key: str) -> str:
    url = "https://dataservice.accuweather.com/locations/v1/cities/geoposition/search"
    params = {"apikey": api_key, "q": f"{location.lat},{location.lon}"}
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()
    try:
        return payload["Key"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unexpected AccuWeather geoposition response structure") from exc


def fetch_forecast(location: Location) -> ForecastRecord:
    """Fetch tomorrow's forecast from AccuWeather's 5-day daily forecast endpoint.

    Requires ACCUWEATHER_KEY in your .env. This collector is live-forecast only;
    historical backfill remains Open-Meteo's job.

    Note: AccuWeather's location-key lookup and daily-forecast field names (in
    particular UV index and humidity, which the base 5-day endpoint may not expose)
    were not verified against a live response while building this - worth
    double-checking raw_json the first time this runs with a real key.
    """
    api_key = os.getenv("ACCUWEATHER_KEY")
    if not api_key:
        raise RuntimeError("ACCUWEATHER_KEY is not set. Add it to .env to enable AccuWeather.")

    location_key = _get_location_key(location, api_key)

    url = f"https://dataservice.accuweather.com/forecasts/v1/daily/5day/{location_key}"
    params = {"apikey": api_key, "metric": "true", "details": "true"}
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()

    try:
        day = payload["DailyForecasts"][1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected AccuWeather forecast response structure") from exc

    day_part = day.get("Day", {})
    temperature = day.get("Temperature", {})
    wind = day_part.get("Wind", {}).get("Speed", {})
    wind_gust = day_part.get("WindGust", {}).get("Speed", {})
    # TotalLiquid covers rain/snow/ice combined; Rain is rain-only. Prefer the former
    # as the closer match to other providers' all-precipitation "precipitation_sum".
    precip = day_part.get("TotalLiquid", day_part.get("Rain", {}))
    humidity = day_part.get("RelativeHumidity", {})

    return ForecastRecord(
        source="accuweather",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=date.fromisoformat(day["Date"][:10]),
        collected_at=datetime.now(),
        max_temp=_to_float(temperature.get("Maximum", {}).get("Value")),
        min_temp=_to_float(temperature.get("Minimum", {}).get("Value")),
        rain_probability=_to_float(day_part.get("PrecipitationProbability")),
        precipitation_sum=_to_float(precip.get("Value") if isinstance(precip, dict) else None),
        wind_speed=_to_float(wind.get("Value")),  # AccuWeather's metric wind speed is already km/h.
        wind_gusts=_to_float(wind_gust.get("Value")),
        cloud_cover=_to_float(day_part.get("CloudCover")),
        humidity=_to_float(humidity.get("Average") if isinstance(humidity, dict) else None),
        pressure_msl=None,  # Not exposed by the base 5-day forecast endpoint.
        weather_code=_to_float(day_part.get("Icon")),
        raw_json=payload,
    )
