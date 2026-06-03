from __future__ import annotations

from datetime import date, datetime, timedelta

import requests

from weather_ensemble.config import Location, TIMEOUT_SECONDS
from weather_ensemble.models import ActualRecord, ForecastRecord

DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
    "uv_index_max",
    "wind_speed_10m_max",
]


def _safe(values: list | None, idx: int) -> float | None:
    if not values or idx >= len(values):
        return None
    return values[idx]


def precipitation_to_rain_probability(precipitation_sum: float | None) -> float | None:
    """Convert observed rain amount into a rough probability-like target for scoring."""
    if precipitation_sum is None:
        return None
    if precipitation_sum >= 5:
        return 90.0
    if precipitation_sum >= 1:
        return 70.0
    if precipitation_sum > 0:
        return 30.0
    return 5.0


def fetch_forecast(location: Location, model: str = "best_match") -> ForecastRecord:
    """Fetch tomorrow's daily forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "daily": ",".join([f for f in DAILY_FIELDS if f != "precipitation_sum"]),
        "timezone": location.timezone,
        "forecast_days": 3,
        "models": model,
    }
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    daily = payload["daily"]
    idx = 1

    return ForecastRecord(
        source=f"open_meteo_{model}",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=date.fromisoformat(daily["time"][idx]),
        collected_at=datetime.now(),
        max_temp=_safe(daily.get("temperature_2m_max"), idx),
        min_temp=_safe(daily.get("temperature_2m_min"), idx),
        rain_probability=_safe(daily.get("precipitation_probability_max"), idx),
        uv_index=_safe(daily.get("uv_index_max"), idx),
        wind_speed=_safe(daily.get("wind_speed_10m_max"), idx),
        raw_json=payload,
    )


def fetch_actual(location: Location, target_date: date) -> ActualRecord:
    """Fetch observed weather for a past date from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "daily": ",".join([f for f in DAILY_FIELDS if f != "precipitation_probability_max"]),
        "timezone": location.timezone,
    }
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    daily = payload["daily"]
    precip = _safe(daily.get("precipitation_sum"), 0)

    return ActualRecord(
        source="open_meteo_archive",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        actual_date=target_date,
        collected_at=datetime.now(),
        max_temp=_safe(daily.get("temperature_2m_max"), 0),
        min_temp=_safe(daily.get("temperature_2m_min"), 0),
        rain_probability=precipitation_to_rain_probability(precip),
        precipitation_sum=precip,
        uv_index=_safe(daily.get("uv_index_max"), 0),
        wind_speed=_safe(daily.get("wind_speed_10m_max"), 0),
        raw_json=payload,
    )


def fetch_historical_forecasts(
    location: Location,
    days_back: int,
    model: str = "best_match",
) -> list[ForecastRecord]:
    """Backfill recent Open-Meteo historical forecasts using the forecast endpoint's past_days."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "daily": ",".join([f for f in DAILY_FIELDS if f not in {"precipitation_sum"}]),
        "timezone": location.timezone,
        "past_days": days_back,
        "forecast_days": 1,
        "models": model,
    }
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    daily = payload["daily"]
    records: list[ForecastRecord] = []

    for idx, date_str in enumerate(daily.get("time", [])):
        forecast_date = date.fromisoformat(date_str)
        if forecast_date >= date.today():
            continue
        records.append(
            ForecastRecord(
                source=f"open_meteo_{model}",
                location_name=location.name,
                lat=location.lat,
                lon=location.lon,
                forecast_date=forecast_date,
                collected_at=datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()).replace(hour=21),
                max_temp=_safe(daily.get("temperature_2m_max"), idx),
                min_temp=_safe(daily.get("temperature_2m_min"), idx),
                rain_probability=_safe(daily.get("precipitation_probability_max"), idx),
                uv_index=_safe(daily.get("uv_index_max"), idx),
                wind_speed=_safe(daily.get("wind_speed_10m_max"), idx),
                raw_json={},
            )
        )
    return records
