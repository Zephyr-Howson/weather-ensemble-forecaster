from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import requests

from weather_ensemble.config import (
    PERIOD_HOURS,
    PERIODS,
    RAIN_THRESHOLD_MM,
    TIMEOUT_SECONDS,
    Location,
    local_today,
)
from weather_ensemble.models import (
    ActualPeriodRecord,
    ActualRecord,
    ForecastPeriodRecord,
    ForecastRecord,
)
from weather_ensemble.retry import get_with_retry

FORECAST_DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
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


def _bucket_hourly_by_period(times: list[str], values: list, target_date: date, agg) -> dict[str, float | None]:
    """Bucket one day's hourly values into PERIOD_HOURS, in the location's own
    local time - Open-Meteo's hourly `time` array is already location-local
    since every request here passes `timezone=location.timezone`, so the hour
    can be read directly off the timestamp string with no conversion.
    """
    buckets: dict[str, list[float]] = {period: [] for period in PERIODS}
    for time_str, value in zip(times, values, strict=False):
        if value is None or not time_str.startswith(target_date.isoformat()):
            continue
        hour = int(time_str[11:13])
        for period, (start, end) in PERIOD_HOURS.items():
            if start <= hour < end:
                buckets[period].append(float(value))
                break
    return {period: agg(vals) if vals else None for period, vals in buckets.items()}


def fetch_forecast_with_periods(
    location: Location, target_date: date, model: str = "best_match"
) -> tuple[ForecastRecord, list[ForecastPeriodRecord]]:
    """Fetch target_date's daily forecast and its overnight/morning/afternoon/
    evening rain split from a single API call.

    precipitation_sum (daily and per period) is summed from the hourly
    `precipitation` array rather than taken from Open-Meteo's own
    `daily.precipitation_sum`, which is computed independently and can disagree
    with the hourly values. Summing one array guarantees the daily total equals
    the sum of its 4 periods. Humidity, cloud cover and pressure are daily means
    of the hourly values; period rain_probability is the hourly maximum.

    forecast_days=3 covers local today and tomorrow (where a live target date
    falls) plus a day's margin, and the target's row is found by matching
    daily.time rather than by position. Raises ValueError if it is missing.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "daily": ",".join(FORECAST_DAILY_FIELDS),
        "hourly": ",".join([*FORECAST_HOURLY_FIELDS, "precipitation", "precipitation_probability"]),
        "timezone": location.timezone,
        "forecast_days": 3,
        "models": model,
    }
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()
    daily = payload["daily"]
    hourly = payload.get("hourly", {})
    try:
        idx = daily["time"].index(target_date.isoformat())
    except ValueError as exc:
        raise ValueError(f"Open-Meteo {model} response has no entry for {target_date.isoformat()}") from exc
    collected_at = datetime.now(UTC).replace(tzinfo=None)

    def hourly_for_target(key: str) -> list:
        times = hourly.get("time", [])
        values = hourly.get(key, [])
        return [value for time_str, value in zip(times, values, strict=False) if time_str.startswith(target_date.isoformat())]

    precip_values = [v for v in hourly_for_target("precipitation") if v is not None]
    precipitation_sum = round(sum(precip_values), 3) if precip_values else None

    forecast = ForecastRecord(
        source=f"open_meteo_{model}",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        forecast_date=target_date,
        collected_at=collected_at,
        max_temp=_safe(daily.get("temperature_2m_max"), idx),
        min_temp=_safe(daily.get("temperature_2m_min"), idx),
        rain_probability=_safe(daily.get("precipitation_probability_max"), idx),
        precipitation_sum=precipitation_sum,
        wind_speed=_safe(daily.get("wind_speed_10m_max"), idx),
        wind_gusts=_safe(daily.get("wind_gusts_10m_max"), idx),
        cloud_cover=_mean(hourly_for_target("cloud_cover")),
        humidity=_mean(hourly_for_target("relative_humidity_2m")),
        pressure_msl=_mean(hourly_for_target("pressure_msl")),
        weather_code=_safe(daily.get("weather_code"), idx),
    )

    times = hourly.get("time", [])
    precip_by_period = _bucket_hourly_by_period(
        times, hourly.get("precipitation", []), target_date, lambda vals: round(sum(vals), 3)
    )
    prob_by_period = _bucket_hourly_by_period(times, hourly.get("precipitation_probability", []), target_date, max)
    periods = [
        ForecastPeriodRecord(
            source=f"open_meteo_{model}",
            location_name=location.name,
            lat=location.lat,
            lon=location.lon,
            forecast_date=target_date,
            period=period,
            collected_at=collected_at,
            precipitation_sum=precip_by_period[period],
            rain_probability=prob_by_period[period],
        )
        for period in PERIODS
    ]
    return forecast, periods


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
        collected_at=datetime.now(UTC).replace(tzinfo=None),
        max_temp=_safe(daily.get("temperature_2m_max"), 0),
        min_temp=_safe(daily.get("temperature_2m_min"), 0),
        precipitation_sum=precip,
        did_rain=_did_rain(precip),
        wind_speed=_safe(daily.get("wind_speed_10m_max"), 0),
        wind_gusts=_safe(daily.get("wind_gusts_10m_max"), 0),
        cloud_cover=_mean(hourly.get("cloud_cover")),
        humidity=_mean(hourly.get("relative_humidity_2m")),
        pressure_msl=_mean(hourly.get("pressure_msl")),
        weather_code=_safe(daily.get("weather_code"), 0),
    )


def fetch_actual_periods(location: Location, target_date: date) -> list[ActualPeriodRecord]:
    """Sub-daily rain ground truth: observed precipitation for target_date,
    summed into each of PERIODS (overnight/morning/afternoon/evening, local
    time), from Open-Meteo's archive API.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": "precipitation",
        "timezone": location.timezone,
    }
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()
    hourly = payload.get("hourly", {})
    collected_at = datetime.now(UTC).replace(tzinfo=None)

    precip_by_period = _bucket_hourly_by_period(
        hourly.get("time", []), hourly.get("precipitation", []), target_date, lambda vals: round(sum(vals), 3)
    )

    return [
        ActualPeriodRecord(
            source="open_meteo_archive",
            location_name=location.name,
            lat=location.lat,
            lon=location.lon,
            actual_date=target_date,
            period=period,
            collected_at=collected_at,
            precipitation_sum=precip_by_period[period],
            did_rain=_did_rain(precip_by_period[period]),
        )
        for period in PERIODS
    ]


def fetch_historical_forecast_with_periods(
    location: Location, days_back: int, model: str = "best_match"
) -> tuple[list[ForecastRecord], list[ForecastPeriodRecord]]:
    """Backfill archived daily model forecasts and their period split for the
    days_back days before the location's local today, from a single API call.

    Tries the Historical Forecast API first and falls back to the live
    forecast API's past_days window. As in fetch_forecast_with_periods, the
    daily and period precipitation totals are both summed from the hourly
    `precipitation` array. Rows are marked collection_method="backfill", with
    collected_at set to 21:00 on the day before each forecast date.

    Period rows carry no rain_probability: hourly probability isn't reliably
    available on these endpoints, and probability is only an ML feature, never
    the regression target.
    """
    today = local_today(location)
    start = today - timedelta(days=days_back)
    end = today - timedelta(days=1)
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(FORECAST_DAILY_FIELDS),
        "hourly": ",".join([*FORECAST_HOURLY_FIELDS, "precipitation"]),
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
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover
            last_error = exc

    if payload is None:
        raise RuntimeError(f"Could not fetch historical forecasts: {last_error}")

    daily = payload["daily"]
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    precip_values = hourly.get("precipitation", [])

    def hourly_for_date(target: date, key: str) -> list:
        values = hourly.get(key, [])
        return [value for time_str, value in zip(times, values, strict=False) if time_str.startswith(target.isoformat())]

    forecasts: list[ForecastRecord] = []
    periods: list[ForecastPeriodRecord] = []
    for idx, date_str in enumerate(daily.get("time", [])):
        forecast_date = date.fromisoformat(date_str)
        if forecast_date >= today:
            continue
        collected_at = datetime.combine(forecast_date - timedelta(days=1), datetime.min.time()).replace(hour=21)

        day_precip = [v for v in hourly_for_date(forecast_date, "precipitation") if v is not None]
        precipitation_sum = round(sum(day_precip), 3) if day_precip else None

        forecasts.append(
            ForecastRecord(
                source=f"open_meteo_{model}",
                location_name=location.name,
                lat=location.lat,
                lon=location.lon,
                forecast_date=forecast_date,
                collected_at=collected_at,
                max_temp=_safe(daily.get("temperature_2m_max"), idx),
                min_temp=_safe(daily.get("temperature_2m_min"), idx),
                rain_probability=_safe(daily.get("precipitation_probability_max"), idx),
                precipitation_sum=precipitation_sum,
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

        precip_by_period = _bucket_hourly_by_period(times, precip_values, forecast_date, lambda vals: round(sum(vals), 3))
        for period in PERIODS:
            periods.append(
                ForecastPeriodRecord(
                    source=f"open_meteo_{model}",
                    location_name=location.name,
                    lat=location.lat,
                    lon=location.lon,
                    forecast_date=forecast_date,
                    period=period,
                    collected_at=collected_at,
                    precipitation_sum=precip_by_period[period],
                    rain_probability=None,
                    collection_method="backfill",
                )
            )
    return forecasts, periods


def fetch_historical_actual_periods(location: Location, days_back: int) -> list[ActualPeriodRecord]:
    """Batched fetch_actual_periods for the days_back days before the
    location's local today: one date-range archive request instead of one
    per day, which keeps a full 30-location backfill fast.
    """
    today = local_today(location)
    start = today - timedelta(days=days_back)
    end = today - timedelta(days=1)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": location.lat,
        "longitude": location.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": "precipitation",
        "timezone": location.timezone,
    }
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    values = hourly.get("precipitation", [])
    collected_at = datetime.now(UTC).replace(tzinfo=None)

    records: list[ActualPeriodRecord] = []
    day = start
    while day <= end:
        precip_by_period = _bucket_hourly_by_period(times, values, day, lambda vals: round(sum(vals), 3))
        for period in PERIODS:
            precip = precip_by_period[period]
            records.append(
                ActualPeriodRecord(
                    source="open_meteo_archive",
                    location_name=location.name,
                    lat=location.lat,
                    lon=location.lon,
                    actual_date=day,
                    period=period,
                    collected_at=collected_at,
                    precipitation_sum=precip,
                    did_rain=_did_rain(precip),
                )
            )
        day += timedelta(days=1)
    return records
