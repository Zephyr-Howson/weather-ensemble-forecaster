# Weather Ensemble Forecaster

A Python-first weather aggregation project that collects forecasts from multiple sources, stores the real weather afterwards, builds a modelling table, and trains an ML post-processing model to improve tomorrow's forecast.

The project is organised around three phases:

1. **Phase 1 — Collect/backfill data**  
   Pull historical forecast archives and actual observations, then keep collecting live forecasts every day.
2. **Phase 2 — Build a modelling dataset**  
   Convert source-level forecasts into a wide feature table suitable for modelling.
3. **Phase 3 — Train and use an ML ensemble**  
   Train one model per weather variable and compare it against the simpler weighted-average baseline.

Open-Meteo is the first-class source because it offers both historical weather observations and historical forecast archives. That means you do not need to wait weeks before training a first model.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Check the CLI works:

```powershell
weather-ensemble --help
```

Run tests:

```powershell
pytest
```

---

## Quick start: deploy all 3 phases locally

For Melbourne:

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --deploy-phases 180
```

This will:

- backfill 180 days of actual observations;
- backfill Open-Meteo historical forecasts;
- collect the latest live forecasts;
- build `data/processed/features.parquet` or fall back to `features.csv`;
- train models into `models/`;
- print the latest ML prediction if enough data exists.

For a quicker smoke test:

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --deploy-phases 30
```

---

## Daily workflow

Once the database exists, run this each night or morning:

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --all --window 14
```

That runs:

- collect tomorrow's forecasts;
- record yesterday's actual weather;
- generate the weighted-average baseline forecast.

Then periodically retrain the ML models:

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --train
```

Generate the ML forecast:

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --predict-ml
```

---

## Useful commands

### Backfill historical forecasts and actuals

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --backfill 365
```

### Build the wide ML feature table

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --build-dataset data/processed/features.parquet
```

### Train ML models

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --train --model-dir models
```

### Predict with ML models

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --predict-ml --model-dir models
```

### Weighted-average baseline forecast

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --forecast --window 14
```

---

## Repository structure

```text
weather-ensemble-forecaster/
├── src/weather_ensemble/
│   ├── cli.py                 # command line interface
│   ├── config.py              # default location, variables, DB path
│   ├── db.py                  # SQLite schema and insert/upsert functions
│   ├── models.py              # dataclasses for forecasts and actuals
│   ├── service.py             # collection, backfill, scoring, baseline blending
│   ├── ml.py                  # Phase 2/3 feature building, training, prediction
│   ├── phases.py              # one-command pipeline for all 3 phases
│   └── sources/
│       ├── open_meteo.py      # live forecast, historical forecast, actual weather
│       └── wttr.py            # live forecast source
├── tests/                     # pytest tests
├── data/                      # local DB/raw/processed data, ignored by git
├── models/                    # trained ML model files, ignored by git
├── legacy/                    # original single-file prototype
├── pyproject.toml             # package metadata and dependencies
└── README.md
```

---

## What the model currently does

The Phase 3 model trains separate `RandomForestRegressor` models for:

- max temperature;
- min temperature;
- rain probability;
- UV index;
- wind speed.

Features include:

- each provider's forecast for each variable;
- cross-source mean, min, max, and standard deviation;
- month;
- day of year;
- day of week.

This is intentionally simple and robust. The next upgrade would be to add:

- hourly forecasts instead of only daily values;
- forecast horizon features;
- previous day's actual weather;
- BOM or other Australian-specific sources;
- XGBoost/LightGBM;
- a proper backtest comparing ML vs best single provider vs simple average vs weighted average.

---

## Data note

There is an important distinction:

- **Historical weather** = what actually happened.
- **Historical forecast** = what a weather model predicted before it happened.

This project needs both. The actuals are the target. The historical forecasts are the inputs.

## Variable set upgraded in v2

The pipeline now separates **forecast features** from **observed targets**.

Forecast features collected where each provider supports them:

- `max_temp`
- `min_temp`
- `rain_probability`
- `precipitation_sum`
- `uv_index`
- `wind_speed`
- `wind_gusts`
- `cloud_cover`
- `humidity`
- `pressure_msl`
- `weather_code`

Observed targets used for model training:

- `actual_max_temp`
- `actual_min_temp`
- `actual_precipitation_sum`
- `actual_did_rain`
- `actual_uv_index`
- `actual_wind_speed`
- `actual_wind_gusts`

`rain_probability` is no longer treated as an actual target, because probability is a forecast concept rather than something directly observed. Verification now uses `did_rain` and `precipitation_sum`.

The wide ML feature table now also includes source-agreement features for each variable:

- `source_mean__<variable>`
- `source_median__<variable>`
- `source_std__<variable>`
- `source_min__<variable>`
- `source_max__<variable>`
- `source_range__<variable>`
- `source_count__<variable>`

These features let the model learn when providers agree, disagree, or have missing data.


## Optional API providers

The project works without API keys using Open-Meteo and wttr.in. To add more live forecast sources, create a `.env` file from `.env.example` and add:

```env
WEATHERAPI_KEY=your_weatherapi_key
VISUAL_CROSSING_KEY=your_visual_crossing_key
```

Then run:

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --collect
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --build-dataset data/processed/features.parquet
```

Notes:

- Open-Meteo remains the main historical forecast backfill source.
- WeatherAPI and Visual Crossing are included as live forecast collectors in this version. Their standard history endpoints are not treated as true archived forecasts in this project.
- Humidity, cloud cover and mean sea-level pressure now flow through the Open-Meteo, WeatherAPI, Visual Crossing and wttr.in collectors when available.


## V4: Free Open-Meteo Multi-Model Ensemble

This version uses Open-Meteo as the core no-key ensemble. Instead of treating
Open-Meteo as one source, it collects several model outputs as separate sources:

- `open_meteo_best_match`
- `open_meteo_ecmwf_ifs025`
- `open_meteo_gfs_global`
- `open_meteo_icon_global`
- `open_meteo_gem_seamless`

This is the recommended first architecture because these sources can be
backfilled using Open-Meteo historical forecasts. WeatherAPI and Visual Crossing
remain optional live sources if API keys are available.

### Run the free full pipeline

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --deploy-phases 365
```

### Collect only the free Open-Meteo models for tomorrow

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --collect-open-meteo
```

### Inspect the wide feature table

```powershell
python -c "import pandas as pd; df=pd.read_parquet('data/processed/features.parquet'); print(df.shape); print(df.filter(like='source_count').head())"
```

You should now see source counts greater than 1, and `source_std__...` /
`source_range__...` features should become meaningful.
