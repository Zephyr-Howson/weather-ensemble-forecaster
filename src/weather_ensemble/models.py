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
    precipitation_sum: float | None = None
    wind_speed: float | None = None
    wind_gusts: float | None = None
    cloud_cover: float | None = None
    humidity: float | None = None
    pressure_msl: float | None = None
    weather_code: float | None = None
    raw_json: dict[str, Any] | None = None
    collection_method: str = "live"


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
    precipitation_sum: float | None = None
    did_rain: int | None = None
    wind_speed: float | None = None
    wind_gusts: float | None = None
    cloud_cover: float | None = None
    humidity: float | None = None
    pressure_msl: float | None = None
    weather_code: float | None = None
    raw_json: dict[str, Any] | None = None

    @property
    def rain_probability(self) -> None:
        """Deprecated compatibility shim.

        Observed rain probability is intentionally not stored anymore. Use
        did_rain and precipitation_sum for verification and model targets.
        """
        return None
