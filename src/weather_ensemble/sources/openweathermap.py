from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from weather_ensemble.config import Location, TIMEOUT_SECONDS, local_today
from weather_ensemble.models import ForecastRecord


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _kmh(value: object) -> float | None:
    """OpenWeatherMap reports wind speed in m/s under units=metric; convert to km/h."""
    ms = _to_float(value)
    return round(ms * 3.6, 3) if ms is not None else None


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _max(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _min(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _local_date_from_timestamp(ts: float, location: Location) -> date:
    """OWM's "dt" is a UTC unix timestamp with no location-local-date equivalent;
    convert it ourselves rather than using the machine's own timezone."""
    try:
        tz = ZoneInfo(location.timezone)
    except ZoneInfoNotFoundError:
        tz = None
    return datetime.fromtimestamp(ts, tz=tz).date()


def fetch_forecast(location: Location) -> ForecastRecord:
    """Fetch tomorrow's forecast from OpenWeatherMap's free 5 Day / 3 Hour Forecast endpoint.

    Requires OPENWEATHERMAP_KEY in your .env. Deliberately uses /data/2.5/forecast
    rather than One Call 3.0: 2.5 is included in every free plan with no card
    required, while One Call 3.0 needs a separate paid "One Call by Call"
    subscription. The tradeoff is resolution - this endpoint is 3-hourly, so
    it's aggregated here into one daily summary - and no UV index, which isn't
    exposed on this endpoint at all.
    """
    api_key = os.getenv("OPENWEATHERMAP_KEY")
    if not api_key:
        raise RuntimeError("OPENWEATHERMAP_KEY is not set. Add it to .env to enable OpenWeatherMap.")

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": location.lat,
        "lon": location.lon,
        "appid": api_key,
        "units": "metric",
    }
    response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    try:
        entries = payload["list"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Unexpected OpenWeatherMap response structure") from exc

    target_date = local_today(location) + timedelta(days=1)
    day_entries = [e for e in entries if _local_date_from_timestamp(e["dt"], location) == target_date]
    if not day_entries:
        raise ValueError(f"No OpenWeatherMap 3-hour entries found for {target_date.isoformat()}")

    main = [e.get("main", {}) for e in day_entries]
    wind = [e.get("wind", {}) for e in day_entries]
    clouds = [e.get("clouds", {}) for e in day_entries]
    pops = [_to_float(e.get("pop")) for e in day_entries]
    # OWM omits the "rain" key entirely on dry 3-hour blocks rather than sending 0,
    # so absent blocks contribute 0 to the daily total (a true zero, not unreported).
    rains = [_to_float((e.get("rain") or {}).get("3h")) or 0.0 for e in day_entries]
    weather_ids = [(e.get("weather") or [{}])[0].get("id") for e in day_entries]

    rain_prob = _max(pops)

    return ForecastRecord(
        source="openweathermap",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=target_date,
        collected_at=datetime.now(),
        max_temp=_max([_to_float(m.get("temp_max")) for m in main]),
        min_temp=_min([_to_float(m.get("temp_min")) for m in main]),
        rain_probability=round(rain_prob * 100, 1) if rain_prob is not None else None,
        precipitation_sum=round(sum(rains), 3),
        uv_index=None,  # Not exposed by the free 3-hourly forecast endpoint.
        wind_speed=_kmh(_max([_to_float(w.get("speed")) for w in wind])),
        wind_gusts=_kmh(_max([_to_float(w.get("gust")) for w in wind])),
        cloud_cover=_mean([_to_float(c.get("all")) for c in clouds]),
        humidity=_mean([_to_float(m.get("humidity")) for m in main]),
        pressure_msl=_mean([_to_float(m.get("sea_level", m.get("pressure"))) for m in main]),
        weather_code=_to_float(weather_ids[len(weather_ids) // 2]) if weather_ids else None,
        raw_json=payload,
    )
