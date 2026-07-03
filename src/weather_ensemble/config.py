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

# Open-Meteo model suite used as the first serious ensemble.
# These are separate numerical weather prediction systems exposed through
# Open-Meteo and can be backfilled through the Historical Forecast API.
OPEN_METEO_MODELS: dict[str, str] = {
    "best_match": "Open-Meteo Best Match",
    "ecmwf_ifs025": "ECMWF IFS 0.25°",
    "gfs_global": "NOAA GFS Global",
    "gem_seamless": "Environment Canada GEM Seamless",
}

# Some Open-Meteo models may return sparse or null fields in certain regions.
# Keep them optional unless they are verified to provide stable data for your
# target locations.
OPTIONAL_OPEN_METEO_MODELS: dict[str, str] = {
    "icon_global": "DWD ICON Global",
    "bom_access_global": "BOM ACCESS Global",
}

# Optional live-only providers. They are useful going forward, but they do not
# give the same historical forecast backfill advantage as Open-Meteo.
OPTIONAL_LIVE_PROVIDERS = ["weatherapi", "visual_crossing", "wttr_in"]



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
