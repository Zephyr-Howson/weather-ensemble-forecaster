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


def _max_hourly(hourly: list[dict], key: str) -> float | None:
    vals = [_to_float(h.get(key)) for h in hourly]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _mean_hourly(hourly: list[dict], key: str) -> float | None:
    vals = [_to_float(h.get(key)) for h in hourly]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


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

    # wttr.in exposes fewer fields and uses different names, so missing values
    # are left as None rather than faked. The ML pipeline can handle this.
    return ForecastRecord(
        source="wttr_in",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=date.fromisoformat(day["date"]),
        collected_at=datetime.now(),
        max_temp=_to_float(day.get("maxtempC")),
        min_temp=_to_float(day.get("mintempC")),
        rain_probability=_max_hourly(hourly, "chanceofrain"),
        precipitation_sum=None,  # wttr JSON lacks reliable daily rain total
        uv_index=_max_hourly(hourly, "uvIndex"),
        wind_speed=_max_hourly(hourly, "windspeedKmph"),
        wind_gusts=_max_hourly(hourly, "WindGustKmph"),
        cloud_cover=_mean_hourly(hourly, "cloudcover"),
        humidity=_mean_hourly(hourly, "humidity"),
        pressure_msl=_mean_hourly(hourly, "pressure"),
        weather_code=_to_float(hourly[0].get("weatherCode")) if hourly else None,
        raw_json=payload,
    )
