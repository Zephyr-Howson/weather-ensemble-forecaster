from __future__ import annotations

from datetime import date, datetime, timedelta

from weather_ensemble.config import Location, RAIN_THRESHOLD_MM, TIMEOUT_SECONDS, local_today
from weather_ensemble.models import ActualRecord, ForecastRecord
from weather_ensemble.retry import get_with_retry

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

# Open-Meteo exposes humidity, cloud cover and pressure as hourly variables.
# We summarise them to daily means so they can be compared across providers.
FORECAST_HOURLY_FIELDS = [
    "relative_humidity_2m",
    "cloud_cover",
    "pressure_msl",
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
        "hourly": ",".join(FORECAST_HOURLY_FIELDS),
        "timezone": location.timezone,
        "forecast_days": 3,
        "models": model,
    }
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()
    daily = payload["daily"]
    hourly = payload.get("hourly", {})
    idx = 1
    target_date = date.fromisoformat(daily["time"][idx])

    def hourly_for_target(key: str) -> list:
        times = hourly.get("time", [])
        values = hourly.get(key, [])
        return [value for time_str, value in zip(times, values, strict=False) if time_str.startswith(target_date.isoformat())]

    return ForecastRecord(
        source=f"open_meteo_{model}",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=target_date,
        collected_at=datetime.now(),
        max_temp=_safe(daily.get("temperature_2m_max"), idx),
        min_temp=_safe(daily.get("temperature_2m_min"), idx),
        rain_probability=_safe(daily.get("precipitation_probability_max"), idx),
        precipitation_sum=_safe(daily.get("precipitation_sum"), idx),
        uv_index=_safe(daily.get("uv_index_max"), idx),
        wind_speed=_safe(daily.get("wind_speed_10m_max"), idx),
        wind_gusts=_safe(daily.get("wind_gusts_10m_max"), idx),
        cloud_cover=_mean(hourly_for_target("cloud_cover")),
        humidity=_mean(hourly_for_target("relative_humidity_2m")),
        pressure_msl=_mean(hourly_for_target("pressure_msl")),
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
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
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
    today = local_today(location)
    start = today - timedelta(days=days_back)
    end = today - timedelta(days=1)
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(FORECAST_DAILY_FIELDS),
        "hourly": ",".join(FORECAST_HOURLY_FIELDS),
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
            response = get_with_retry(url, params=request_params, timeout=TIMEOUT_SECONDS)
            payload = response.json()
            break
        except Exception as exc:  # pragma: no cover
            last_error = exc

    if payload is None:
        raise RuntimeError(f"Could not fetch historical forecasts: {last_error}")

    daily = payload["daily"]
    hourly = payload.get("hourly", {})

    def hourly_for_date(target: date, key: str) -> list:
        times = hourly.get("time", [])
        values = hourly.get(key, [])
        return [value for time_str, value in zip(times, values, strict=False) if time_str.startswith(target.isoformat())]

    records: list[ForecastRecord] = []
    for idx, date_str in enumerate(daily.get("time", [])):
        forecast_date = date.fromisoformat(date_str)
        if forecast_date >= today:
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
                cloud_cover=_mean(hourly_for_date(forecast_date, "cloud_cover")),
                humidity=_mean(hourly_for_date(forecast_date, "relative_humidity_2m")),
                pressure_msl=_mean(hourly_for_date(forecast_date, "pressure_msl")),
                weather_code=_safe(daily.get("weather_code"), idx),
                raw_json={"endpoint": "historical_forecast_or_past_days", "model": model},
                collection_method="backfill",
            )
        )
    return records
