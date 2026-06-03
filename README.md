# Weather Ensemble Forecaster

A Python-first weather aggregation project that collects forecasts from multiple sources, records actual observed weather, scores each source, and blends tomorrow's forecast using inverse-error weighting.

This repo is designed as the backend prototype for a future weather app.

## What it does

- Collects tomorrow's daily forecast from:
  - Open-Meteo `best_match`
  - Open-Meteo `bom_access_global`
  - wttr.in
- Records observed actuals from Open-Meteo Archive.
- Backfills recent Open-Meteo forecast and actual history so you do not need to wait weeks before testing.
- Stores data in local SQLite.
- Exports a modelling table to Parquet.
- Produces a simple weighted ensemble forecast.

## Repository structure

```text
weather-ensemble-forecaster/
  src/weather_ensemble/
    cli.py                  # command-line entry point
    config.py               # env/config/location settings
    db.py                   # SQLite schema and inserts
    models.py               # typed records
    service.py              # orchestration, scoring, blending
    sources/
      open_meteo.py         # Open-Meteo forecast/archive clients
      wttr.py               # wttr.in client
  tests/
  scripts/
  data/
    raw/
    processed/
  .github/workflows/tests.yml
  .env.example
  pyproject.toml
  requirements.txt
```

## Setup

```bash
git clone <your-repo-url>
cd weather-ensemble-forecaster
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
cp .env.example .env
```

Edit `.env` if you want to change the default location.

## First-time run

Backfill 30 days of Open-Meteo forecasts and actuals:

```bash
weather-ensemble --backfill 30
```

Collect fresh forecasts, record yesterday's actuals, and generate tomorrow's ensemble:

```bash
weather-ensemble --all
```

For Melbourne explicitly:

```bash
weather-ensemble --name Melbourne --lat -37.8136 --lon 144.9631 --timezone Australia/Melbourne --all
```

Export a modelling table:

```bash
weather-ensemble --export data/processed/modelling_table.parquet
```

Run tests:

```bash
pytest -q
```

## Data model

### `forecasts`

One row per source, location, target forecast date, and collection time.

Key fields:

- `source`
- `location_name`
- `forecast_date`
- `collected_at`
- `max_temp`
- `min_temp`
- `rain_probability`
- `uv_index`
- `wind_speed`
- `raw_json`

### `actuals`

One row per source, location, and actual date.

Key fields:

- `source`
- `location_name`
- `actual_date`
- `collected_at`
- `max_temp`
- `min_temp`
- `rain_probability`
- `precipitation_sum`
- `uv_index`
- `wind_speed`
- `raw_json`

## Collection frequency

Recommended schedule while prototyping:

- Forecast snapshot: once daily between 8pm and 10pm local time.
- Actual weather: once daily the following morning or evening.
- Backfill: once when setting up a new location.
- Model/ensemble update: after actuals are recorded.

On macOS/Linux cron example:

```cron
0 21 * * * cd /path/to/weather-ensemble-forecaster && .venv/bin/weather-ensemble --collect
0 8 * * * cd /path/to/weather-ensemble-forecaster && .venv/bin/weather-ensemble --record-actual --forecast
```

## What to collect next

The current repo starts with daily-level features. The next useful upgrade is hourly data:

- hourly temperature
- hourly rain probability
- precipitation amount
- wind speed and gusts
- humidity
- pressure
- cloud cover
- weather code

For modelling, start with a simple benchmark:

1. Equal-weight average of all sources.
2. Inverse-MAE weighted average.
3. Ridge regression or RandomForest using each source's forecast as features.

Only keep the ML model if it beats the simple average on a rolling validation window.
