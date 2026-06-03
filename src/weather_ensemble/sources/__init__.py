from __future__ import annotations

from collections.abc import Callable

from weather_ensemble.config import Location
from weather_ensemble.models import ForecastRecord
from weather_ensemble.sources import open_meteo, wttr

ForecastFetcher = Callable[[Location], ForecastRecord]

FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    "open_meteo_best_match": lambda location: open_meteo.fetch_forecast(location, "best_match"),
    "open_meteo_bom_access_global": lambda location: open_meteo.fetch_forecast(location, "bom_access_global"),
    "wttr_in": wttr.fetch_forecast,
}
