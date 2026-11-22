from __future__ import annotations

from collections.abc import Callable
from datetime import date

from weather_ensemble.config import OPEN_METEO_MODELS, Location
from weather_ensemble.models import ForecastRecord
from weather_ensemble.sources import (
    accuweather,
    bom,
    open_meteo,
    openweathermap,
    visual_crossing,
    weatherapi,
    weatherbit,
    wttr,
)

# Every fetcher takes the exact calendar date (location-local) it must return a
# forecast for. The caller resolves that date once per location per run (see
# cli._run_for_location and service.default_forecast_target_date); fetchers never
# work out "tomorrow" themselves, so a run that starts late enough to cross local
# midnight can't silently collect the wrong day. A fetcher whose response has no
# entry for the requested date raises, and service.collect_forecasts logs the
# failure and carries on with the other sources.
ForecastFetcher = Callable[[Location, date], ForecastRecord]

# Core free ensemble: one source per Open-Meteo model. These matter most because
# they can be collected live and (except best_match) backfilled from Open-Meteo's
# historical-forecast archive - see config.OPEN_METEO_BACKFILL_MODELS.
OPEN_METEO_FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    f"open_meteo_{model}": (
        lambda location, target_date, model=model: open_meteo.fetch_forecast_with_periods(
            location, target_date, model
        )[0]
    )
    for model in OPEN_METEO_MODELS
}

# Live-only providers (no archived forecasts to backfill from). Keyed ones raise
# when their API key is unset; service.collect_forecasts logs any failure and
# skips that source for the run.
OPTIONAL_FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    "wttr_in": wttr.fetch_forecast,
    "weatherapi": weatherapi.fetch_forecast,
    "visual_crossing": visual_crossing.fetch_forecast,
    "openweathermap": openweathermap.fetch_forecast,
    "weatherbit": weatherbit.fetch_forecast,
    "accuweather": accuweather.fetch_forecast,
    "bom": bom.fetch_forecast,
}

FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    **OPEN_METEO_FORECAST_SOURCES,
    **OPTIONAL_FORECAST_SOURCES,
}
