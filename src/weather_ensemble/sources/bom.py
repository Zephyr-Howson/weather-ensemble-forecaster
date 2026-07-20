from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from weather_ensemble.config import Location, TIMEOUT_SECONDS
from weather_ensemble.models import ForecastRecord
from weather_ensemble.retry import get_with_retry


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _local_date(iso_timestamp: str, timezone: str) -> date:
    dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    return dt.astimezone(ZoneInfo(timezone)).date()


def _geohash(location: Location) -> str:
    url = "https://api.weather.bom.gov.au/v1/locations"
    response = get_with_retry(url, params={"search": f"{location.lat},{location.lon}"}, timeout=TIMEOUT_SECONDS)
    payload = response.json()
    try:
        return payload["data"][0]["geohash"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected BOM location search response structure") from exc


def fetch_forecast(location: Location) -> ForecastRecord:
    """Fetch tomorrow's daily forecast from BOM's public API.

    This hits api.weather.bom.gov.au, the JSON API behind bom.gov.au's own forecast
    pages. It is reverse-engineered, not an officially documented or supported
    integration - its own responses carry a "must not use, copy or share" notice,
    so treat this as a best-effort personal/non-commercial source that could change
    or get blocked without notice, unlike the other providers here.

    BOM's daily forecast has no wind/humidity/cloud/pressure numerics and no WMO
    weather_code - only a rain chance/amount range, UV index, and temp max/min -
    so those fields are left None, same as silo.py.
    """
    geohash = _geohash(location)
    url = f"https://api.weather.bom.gov.au/v1/locations/{geohash}/forecasts/daily"
    response = get_with_retry(url, timeout=TIMEOUT_SECONDS)
    payload = response.json()

    try:
        day = payload["data"][1]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected BOM forecast response structure") from exc

    rain = day.get("rain", {})
    amount = rain.get("amount", {})
    amount_min = _to_float(amount.get("min"))
    amount_max = _to_float(amount.get("max"))
    if amount_min is not None and amount_max is not None:
        precipitation_sum = round((amount_min + amount_max) / 2, 3)
    else:
        precipitation_sum = amount_max if amount_max is not None else amount_min

    return ForecastRecord(
        source="bom",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=_local_date(day["date"], location.timezone),
        collected_at=datetime.now(),
        max_temp=_to_float(day.get("temp_max")),
        min_temp=_to_float(day.get("temp_min")),
        rain_probability=_to_float(rain.get("chance")),
        precipitation_sum=precipitation_sum,
        raw_json=payload,
    )
