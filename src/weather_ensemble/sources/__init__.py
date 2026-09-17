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

# Every fetcher takes the exact calendar date it must return a forecast for -
# resolved once per location by the caller (see service.collect_forecasts),
# not recomputed independently inside each fetcher. Before this, every
# fetcher computed "tomorrow" itself (via local_today(location) or each
# provider's own server-side date), with no protection against a delayed run
# - unlike the ensemble/ML/Best layer (see default_forecast_target_date),
# which already guards against exactly this. A real incident: GitHub's
# scheduled trigger for the daily collection workflow ran 5-12 hours late
# every day from 2026-08-27 onward, late enough that many locations' fetches
# executed past their own local midnight, silently shifting "tomorrow" a full
# day ahead and leaving the intended date's row missing forever for every
# source with no historical-backfill path. Threading one already-resolved
# date through every fetcher closes that gap: a fetcher that can't find data
# for the requested date now raises (caught by collect_forecasts's existing
# per-source try/except) instead of silently substituting a different date.
ForecastFetcher = Callable[[Location, date], ForecastRecord]

# Core free ensemble: multiple Open-Meteo model outputs. These are the most
# important sources because they can be both collected live and backfilled.
OPEN_METEO_FORECAST_SOURCES: dict[str, ForecastFetcher] = {
    f"open_meteo_{model}": (
        lambda location, target_date, model=model: open_meteo.fetch_forecast_with_periods(
            location, target_date, model
        )[0]
    )
    for model in OPEN_METEO_MODELS
}

# Optional live-only providers. These enrich forecasts going forward if API keys
# are configured, but the service layer will skip them cleanly on failures.
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
