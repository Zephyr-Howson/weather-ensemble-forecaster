from __future__ import annotations

from datetime import date, datetime, timedelta

import requests

from weather_ensemble.config import Location, RAIN_THRESHOLD_MM, TIMEOUT_SECONDS
from weather_ensemble.models import ActualRecord, ForecastRecord

FORECAST_DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
    "uv_index_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "weather_code",
]

# Daily archive does not expose all daily means consistently, so cloud/humidity/
# pressure are requested hourly and summarised into daily means.
ACTUAL_DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "uv_index_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "weather_code",
]
ACTUAL_HOURLY_FIELDS = [
    "relative_humidity_2m",
    "cloud_cover",
    "pressure_msl",
]


def _safe(values: list | None, idx: int) -> float | None:
    if not values or idx >= len(values):
        return None
    return values[idx]


def _mean(values: list | None) -> float | None:
    if not values:
        return None
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _did_rain(precipitation_sum: float | None) -> int | None:
    if precipitation_sum is None:
        return None
    return int(precipitation_sum >= RAIN_THRESHOLD_MM)


def fetch_forecast(location: Location, model: str = "best_match") -> ForecastRecord:
    """Fetch tomorrow's daily forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "daily": ",".join(FORECAST_DAILY_FIELDS),
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
        precipitation_sum=_safe(daily.get("precipitation_sum"), idx),
        uv_index=_safe(daily.get("uv_index_max"), idx),
        wind_speed=_safe(daily.get("wind_speed_10m_max"), idx),
        wind_gusts=_safe(daily.get("wind_gusts_10m_max"), idx),
        weather_code=_safe(daily.get("weather_code"), idx),
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
        "daily": ",".join(ACTUAL_DAILY_FIELDS),
        "hourly": ",".join(ACTUAL_HOURLY_FIELDS),
        "timezone": location.timezone,
    }
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()
    daily = payload["daily"]
    hourly = payload.get("hourly", {})
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
        precipitation_sum=precip,
        did_rain=_did_rain(precip),
        uv_index=_safe(daily.get("uv_index_max"), 0),
        wind_speed=_safe(daily.get("wind_speed_10m_max"), 0),
        wind_gusts=_safe(daily.get("wind_gusts_10m_max"), 0),
        cloud_cover=_mean(hourly.get("cloud_cover")),
        humidity=_mean(hourly.get("relative_humidity_2m")),
        pressure_msl=_mean(hourly.get("pressure_msl")),
        weather_code=_safe(daily.get("weather_code"), 0),
        raw_json=payload,
    )


def fetch_historical_forecasts(location: Location, days_back: int, model: str = "best_match") -> list[ForecastRecord]:
    """Backfill archived model forecasts, not observations."""
    start = date.today() - timedelta(days=days_back)
    end = date.today() - timedelta(days=1)
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(FORECAST_DAILY_FIELDS),
        "timezone": location.timezone,
        "models": model,
    }

    endpoints = [
        "https://historical-forecast-api.open-meteo.com/v1/forecast",
        "https://api.open-meteo.com/v1/forecast",
    ]

    payload = None
    last_error: Exception | None = None
    for url in endpoints:
        request_params = dict(params)
        if "api.open-meteo.com" in url:
            request_params.pop("start_date", None)
            request_params.pop("end_date", None)
            request_params["past_days"] = days_back
            request_params["forecast_days"] = 1
        try:
            response = requests.get(url, params=request_params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if payload is None:
        raise RuntimeError(f"Could not fetch historical forecasts: {last_error}")

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
                precipitation_sum=_safe(daily.get("precipitation_sum"), idx),
                uv_index=_safe(daily.get("uv_index_max"), idx),
                wind_speed=_safe(daily.get("wind_speed_10m_max"), idx),
                wind_gusts=_safe(daily.get("wind_gusts_10m_max"), idx),
                weather_code=_safe(daily.get("weather_code"), idx),
                raw_json={"endpoint": "historical_forecast_or_past_days", "model": model},
            )
        )
    return records
