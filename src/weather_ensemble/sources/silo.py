from __future__ import annotations

import os
from datetime import UTC, date, datetime

from weather_ensemble.config import RAIN_THRESHOLD_MM, TIMEOUT_SECONDS, Location
from weather_ensemble.models import ActualRecord
from weather_ensemble.retry import get_with_retry


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _did_rain(precipitation_sum: float | None) -> int | None:
    if precipitation_sum is None:
        return None
    return int(precipitation_sum >= RAIN_THRESHOLD_MM)


def fetch_actual(location: Location, target_date: date) -> ActualRecord:
    """Fetch observed weather for a past date from SILO's gridded DataDrill dataset.

    Requires SILO_EMAIL in your .env (SILO's grid API only needs an email address
    for usage tracking, not a formal API key - see longpaddock.qld.gov.au/silo).
    SILO is an Australia-only, government-run, gridded daily climate dataset built
    from BOM's own station network, going back to 1889 - a genuinely independent
    ground-truth source, not a repackaging of the same reanalysis Open-Meteo uses.

    Unlike the other optional providers, this is NOT wired into the default
    collection pipeline (service.record_actual/backfill): the actuals table
    allows multiple sources per day, but load_modelling_table's forecast-to-
    actual join assumes exactly one actuals row per (location, date) - adding a
    second source there would silently duplicate every forecast row in the
    join. Call this directly if you want to compare or switch to it
    deliberately (own testing found Open-Meteo's own actuals more reliable for
    this project's purposes).
    """
    email = os.getenv("SILO_EMAIL")
    if not email:
        raise RuntimeError("SILO_EMAIL is not set. Add it to .env to enable SILO (just needs an email address).")

    date_str = target_date.strftime("%Y%m%d")
    url = "https://www.longpaddock.qld.gov.au/cgi-bin/silo/DataDrillDataset.php"
    params = {
        "lat": location.lat,
        "lon": location.lon,
        "start": date_str,
        "finish": date_str,
        "format": "json",
        "comment": "RXN",  # R=rainfall, X=max temp, N=min temp
        "username": email,
        "password": "apirequest",
    }
    response = get_with_retry(url, params=params, timeout=TIMEOUT_SECONDS)
    payload = response.json()

    try:
        day = payload["data"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected SILO response structure") from exc

    values = {v.get("variable_code"): v.get("value") for v in day.get("variables", [])}
    precip = _to_float(values.get("daily_rain", values.get("rain")))

    return ActualRecord(
        source="silo",
        location_name=location.name,
        lat=location.lat,
        lon=location.lon,
        actual_date=target_date,
        collected_at=datetime.now(UTC).replace(tzinfo=None),
        max_temp=_to_float(values.get("max_temp")),
        min_temp=_to_float(values.get("min_temp")),
        precipitation_sum=precip,
        did_rain=_did_rain(precip),
        raw_json=payload,
    )
