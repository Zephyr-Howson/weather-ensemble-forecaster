from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class ForecastRecord:
    source: str
    location_name: str
    lat: float
    lon: float
    forecast_date: date
    collected_at: datetime
    max_temp: float | None = None
    min_temp: float | None = None
    rain_probability: float | None = None
    uv_index: float | None = None
    wind_speed: float | None = None
    raw_json: dict[str, Any] | None = None


@dataclass(frozen=True)
class ActualRecord:
    source: str
    location_name: str
    lat: float
    lon: float
    actual_date: date
    collected_at: datetime
    max_temp: float | None = None
    min_temp: float | None = None
    rain_probability: float | None = None
    precipitation_sum: float | None = None
    uv_index: float | None = None
    wind_speed: float | None = None
    raw_json: dict[str, Any] | None = None
