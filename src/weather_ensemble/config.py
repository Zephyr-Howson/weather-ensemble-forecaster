from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    "cloud_cover": "Actual cloud cover (%)",
    "humidity": "Actual relative humidity (%)",
    "pressure_msl": "Actual mean sea-level pressure (hPa)",
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

# Open-Meteo's historical-forecast API does not archive a distinct run for
# "best_match" - it silently returns the same values as the actuals archive.
# Backfilling it would leak the answer into training and let it win a fake
# perfect accuracy score, so only genuinely distinct model runs are backfilled.
# best_match remains a live-only forecast source (see OPEN_METEO_MODELS).
OPEN_METEO_BACKFILL_MODELS: dict[str, str] = {
    key: value for key, value in OPEN_METEO_MODELS.items() if key != "best_match"
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
OPTIONAL_LIVE_PROVIDERS = [
    "weatherapi",
    "visual_crossing",
    "wttr_in",
    "openweathermap",
    "weatherbit",
    "accuweather",
    "bom",
]



@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float
    timezone: str = "auto"


# Curated set of Australian locations for multi-location collection: the 8 state/
# territory capitals, a spread of large regional towns, and popular tourist
# destinations (including Lorne and Shoreham specifically).
AUSTRALIAN_LOCATIONS: list[Location] = [
    # Capital cities
    Location("Sydney", -33.8688, 151.2093, "Australia/Sydney"),
    Location("Melbourne", -37.8136, 144.9631, "Australia/Melbourne"),
    Location("Brisbane", -27.4698, 153.0251, "Australia/Brisbane"),
    Location("Perth", -31.9505, 115.8605, "Australia/Perth"),
    Location("Adelaide", -34.9285, 138.6007, "Australia/Adelaide"),
    Location("Hobart", -42.8821, 147.3272, "Australia/Hobart"),
    Location("Darwin", -12.4634, 130.8456, "Australia/Darwin"),
    Location("Canberra", -35.2809, 149.1300, "Australia/Sydney"),
    # Large regional towns
    Location("Gold Coast", -28.0167, 153.4000, "Australia/Brisbane"),
    Location("Newcastle", -32.9283, 151.7817, "Australia/Sydney"),
    Location("Geelong", -38.1499, 144.3617, "Australia/Melbourne"),
    Location("Cairns", -16.9186, 145.7781, "Australia/Brisbane"),
    Location("Townsville", -19.2590, 146.8169, "Australia/Brisbane"),
    Location("Ballarat", -37.5622, 143.8503, "Australia/Melbourne"),
    Location("Bendigo", -36.7570, 144.2794, "Australia/Melbourne"),
    Location("Launceston", -41.4332, 147.1441, "Australia/Hobart"),
    Location("Wollongong", -34.4278, 150.8931, "Australia/Sydney"),
    Location("Toowoomba", -27.5598, 151.9507, "Australia/Brisbane"),
    Location("Alice Springs", -23.6980, 133.8807, "Australia/Darwin"),
    Location("Mount Gambier", -37.8284, 140.7804, "Australia/Adelaide"),
    # Tourist locations
    Location("Lorne", -38.5423, 143.9750, "Australia/Melbourne"),
    Location("Shoreham", -38.3833, 145.0833, "Australia/Melbourne"),
    Location("Byron Bay", -28.6474, 153.6020, "Australia/Sydney"),
    Location("Port Douglas", -16.4850, 145.4650, "Australia/Brisbane"),
    Location("Broome", -17.9614, 122.2359, "Australia/Perth"),
    Location("Margaret River", -33.9550, 115.0750, "Australia/Perth"),
    Location("Yulara", -25.2406, 130.9889, "Australia/Darwin"),
    Location("Noosa Heads", -26.3936, 153.0919, "Australia/Brisbane"),
    Location("Phillip Island", -38.4495, 145.2410, "Australia/Melbourne"),
    Location("Kangaroo Island", -35.6580, 137.6180, "Australia/Adelaide"),
]


def local_today(location: Location) -> date:
    """The location's current local calendar date, not the machine's.

    A server/CI runner's system clock (typically UTC) can be a different calendar day than the
    forecast location for several hours a day, which would silently shift what "today"/"yesterday"/
    "tomorrow" resolve to. Location.timezone defaults to "auto" (an Open-Meteo API convention meaning
    "the server infers timezone from lat/lon"), which zoneinfo can't resolve directly, so fall back to
    the machine's local date in that case.
    """
    try:
        return datetime.now(ZoneInfo(location.timezone)).date()
    except ZoneInfoNotFoundError:
        return date.today()


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
    return int(os.getenv("ROLLING_WINDOW_DAYS", "30"))
