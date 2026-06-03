from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VARIABLES: dict[str, str] = {
    "max_temp": "Max temperature (°C)",
    "min_temp": "Min temperature (°C)",
    "rain_probability": "Rain probability (%)",
    "uv_index": "UV index",
    "wind_speed": "Wind speed (km/h)",
}

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
