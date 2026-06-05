from __future__ import annotations

from collections.abc import Callable

from weather_ensemble.config import Location, OPEN_METEO_MODELS
from weather_ensemble.models import ForecastRecord
from weather_ensemble.sources import open_meteo, visual_crossing, weatherapi, wttr

ForecastFetcher = Callable[[Location], ForecastRecord]

# Core free ensemble: multiple Open-Meteo model outputs. These are the most
# important sources because they can be both collected live and backfilled.
OPEN_METEO_FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    f"open_meteo_{model}": (lambda location, model=model: open_meteo.fetch_forecast(location, model))
    for model in OPEN_METEO_MODELS
}

# Optional live-only providers. These enrich forecasts going forward if API keys
# are configured, but the service layer will skip them cleanly on failures.
OPTIONAL_FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    "wttr_in": wttr.fetch_forecast,
    "weatherapi": weatherapi.fetch_forecast,
    "visual_crossing": visual_crossing.fetch_forecast,
}

FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    **OPEN_METEO_FORECAST_SOURCES,
    **OPTIONAL_FORECAST_SOURCES,
}
