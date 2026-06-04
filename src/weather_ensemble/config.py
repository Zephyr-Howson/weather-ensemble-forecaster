from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Forecast variables collected from providers. These are inputs/features.
FORECAST_VARIABLES: dict[str, str] = {
    "max_temp": "Max temperature (°C)",
    "min_temp": "Min temperature (°C)",
    "rain_probability": "Rain probability (%)",
    "precipitation_sum": "Forecast precipitation amount (mm)",
    "uv_index": "UV index",
    "wind_speed": "Wind speed (km/h)",
    "wind_gusts": "Wind gusts (km/h)",
    "cloud_cover": "Cloud cover (%)",
    "humidity": "Relative humidity (%)",
    "pressure_msl": "Mean sea-level pressure (hPa)",
    "weather_code": "Weather condition code",
}

# Backwards-compatible alias used by older modules/tests.
VARIABLES = FORECAST_VARIABLES

# True observed targets. Rain probability is deliberately not included because
# probability is a forecast concept; observations should be amount and did-rain.
TARGETS: dict[str, str] = {
    "max_temp": "Actual max temperature (°C)",
    "min_temp": "Actual min temperature (°C)",
    "precipitation_sum": "Actual precipitation amount (mm)",
    "did_rain": "Whether it rained",
    "uv_index": "Actual UV index",
    "wind_speed": "Actual wind speed (km/h)",
    "wind_gusts": "Actual wind gusts (km/h)",
}

RAIN_THRESHOLD_MM = float(os.getenv("RAIN_THRESHOLD_MM", "0.2"))
TIMEOUT_SECONDS = 15


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    timezone: str = "auto"


def get_db_path() -> Path:
    return Path(os.getenv("WEATHER_DB_PATH", "data/weather.db"))


def get_default_location() -> Location:
    return Location(
        name=os.getenv("DEFAULT_LOCATION_NAME", "Melbourne"),
        lat=float(os.getenv("DEFAULT_LAT", "-37.8136")),
        lon=float(os.getenv("DEFAULT_LON", "144.9631")),
        timezone=os.getenv("DEFAULT_TIMEZONE", "Australia/Melbourne"),
    )


def get_rolling_window_days() -> int:
    return int(os.getenv("ROLLING_WINDOW_DAYS", "14"))
