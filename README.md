# Weather Ensemble Forecaster

A daily-automated pipeline that collects tomorrow's forecast from up to twelve
independent weather providers across 30 Australian locations, blends them into
a weighted-average ("Weighted") forecast, trains a separate per-location
machine-learning model as a second, independent prediction, and scores both
against what actually happened. An interactive HTML accuracy dashboard is
regenerated every night showing how the weighted blend, the ML model, every
individual provider, and two naive baselines compare over time.

---

## How the pipeline fits together

Every night, for each of the 30 configured locations:

1. **Collect** — fetch tomorrow's forecast from every configured source (a
   source that fails is logged and skipped; it does not stop the others).
2. **Record actual** — fetch yesterday's *observed* weather (the ground truth
   the whole project is scored against).
3. **Blend** — combine every source's forecast into one **weighted** forecast,
   weighting each source by the inverse of its own recent mean absolute error
   (a source that's been more accurate recently gets more say).
4. **Train** — refit that location's Ridge/Logistic-Regression models on
   everything collected so far.
5. **Predict (ML)** — produce that location's ML forecast for tomorrow.
6. Separately, on request: **backtest** (walk-forward re-creates what the
   weighted blend and ML model *would have* predicted on past dates, using
   only data available before that date — see below), **dedupe** (clean up
   duplicate rows), and **accuracy-report** (regenerate the dashboard).

A single location failing at any step never takes down the other 29 — each
step is isolated per-location, and the run only hard-fails if *every*
location fails (a sign of something systemic, like an expired key, rather
than one provider having a bad night).

---

## Forecast sources

Every source implements `fetch_forecast(location) -> ForecastRecord`, always
for **tomorrow** relative to when it's called (this project has no same-day
or multi-day-ahead forecasting — everything is a 1-day-ahead prediction). The
table below reflects what's actually wired into `sources/__init__.py` today.

| Source (`source` column) | What it is | Cost / key | Backfillable? |
|---|---|---|---|
| `open_meteo_best_match` | Open-Meteo's auto-selected "best" model for the location | Free, no key | **No** — Open-Meteo's historical-forecast API silently returns the *actual* observation for `best_match` instead of an archived forecast, which would leak the answer into training. Live-collected only. |
| `open_meteo_ecmwf_ifs025` | ECMWF IFS 0.25° — the European medium-range model | Free, no key | Yes |
| `open_meteo_gfs_global` | NOAA GFS Global — the US model | Free, no key | Yes |
| `open_meteo_gem_seamless` | Environment Canada GEM Seamless | Free, no key | Yes |
| `wttr_in` | wttr.in (console-friendly weather service) | Free, no key | No (no historical endpoint implemented) |
| `weatherapi` | WeatherAPI.com | Free tier, needs `WEATHERAPI_KEY` | No — its history endpoint returns past observations, not archived forecasts, so it isn't a substitute for real backfill |
| `visual_crossing` | Visual Crossing Timeline API | Free tier, needs `VISUAL_CROSSING_KEY` | No |
| `openweathermap` | OpenWeatherMap's free `/data/2.5/forecast` (3-hourly, aggregated to daily here) | Free tier, needs `OPENWEATHERMAP_KEY` (OpenWeatherMap requires a card on file even for the free tier) | No |
| `weatherbit` | Weatherbit.io | Free tier, needs `WEATHERBIT_KEY` | No |
| `accuweather` | AccuWeather (two-step: geoposition lookup, then 5-day forecast) | Free tier, needs `ACCUWEATHER_KEY` | No |
| `bom` | Australia's Bureau of Meteorology, via its unofficial public JSON API (`api.weather.bom.gov.au`) | Free, no key | No — reverse-engineered, not an officially documented integration. Its own responses carry a "must not use, copy or share" notice, so treat it as best-effort/personal use, not a production dependency. |

**Every field a source can report:**
`max_temp`, `min_temp`, `rain_probability`, `precipitation_sum`,
`wind_speed`, `wind_gusts`, `cloud_cover`, `humidity`, `pressure_msl`,
`weather_code`. Not every source reports every field — gaps are left `NULL`
rather than faked:

- `wttr_in` has no reliable daily rain **total** (only a rain **chance**), so
  `precipitation_sum` is always `None`.
- `accuweather` doesn't expose mean sea-level pressure.
- `visual_crossing` reports conditions as text/icons, not WMO codes, so
  `weather_code` is always `None`.
- `bom`'s daily forecast has no wind/humidity/cloud/pressure/weather-code
  numerics at all — only rain chance/amount and temp max/min.

**UV index was tracked here once and removed.** Every forecast source's
`uv_index` (including WeatherAPI's own forecast) ran systematically 2-4x
higher than the observed ground truth across thousands of scored rows — not
noise, a consistent one-directional bias (forecast APIs commonly report a
clear-sky/theoretical-maximum UV index, while an observed value reflects
actual cloud cover). Averaging several sources that share the same bias
doesn't cancel it out, so the weighted blend scored *worse* than a naive
persistence baseline — a sign the forecast and actual data were measuring
different things, not that the models were bad. Removed rather than kept as
a known-broken metric.

**Configured but not currently active:** `open_meteo_icon_global` (DWD ICON
Global) and `open_meteo_bom_access_global` (BOM ACCESS Global via Open-Meteo)
are listed in `config.py`'s `OPTIONAL_OPEN_METEO_MODELS` but aren't wired into
`sources/__init__.py` yet. `visual_crossing` and `accuweather` are wired in
but currently produce zero rows in the live database — almost certainly a
missing/invalid API key rather than a code issue, since every other source
collects successfully.

### Actuals (the ground truth)

- **`open_meteo_archive`** (via Open-Meteo's Historical Weather API) is the
  default and only actively-used actuals source, called by both
  `record_actual` (yesterday, every night) and `backfill` (a whole date range
  at once). Reports `max_temp`, `min_temp`, `precipitation_sum`, `did_rain`
  (derived from `precipitation_sum >= RAIN_THRESHOLD_MM`, default 0.2mm),
  `wind_speed`, `wind_gusts`, `cloud_cover`, `humidity`, `pressure_msl`,
  `weather_code`.
- **`silo`** (`sources/silo.py`) — Australia-only, government-run, gridded
  daily climate data built from BOM's own station network, going back to
  1889; a genuinely independent ground truth, not a repackaging of the same
  reanalysis Open-Meteo uses. Needs only `SILO_EMAIL` (an email address for
  usage tracking, not a formal API key). **Not wired into the automated
  pipeline** — the `actuals` table technically allows more than one source
  per day, but `load_modelling_table`'s forecast-to-actual join assumes
  exactly one actuals row per (location, date); adding a second source there
  would silently duplicate every forecast row in the join. This project tried
  blending SILO in as a per-field override for a while and reverted it -
  Open-Meteo's own actuals proved more reliable here. Call `silo.fetch_actual`
  directly if you want to compare against it or switch to it deliberately.

### Reliability: retry with backoff

Every source's HTTP call goes through `retry.get_with_retry` (`retry.py`),
which retries a request up to 4 times with exponential backoff (2s, 4s, 8s,
16s) on a connection/timeout error or a `429`/`5xx` response — the classic
"try again shortly" failures — but never retries a `4xx` client error (bad
request, invalid key, not found), since retrying can't fix those. This exists
because a burst of ~120 requests (30 locations × 4 Open-Meteo models) in quick
succession has intermittently hit exactly this kind of transient failure
against Open-Meteo's free tier; each retry is scoped to that one request only,
not a wider re-fetch.

---

## The two predictions

### Weighted (the ensemble blend)

For each forecast variable, every source's value is averaged, weighted by
`1 / that source's own recent mean absolute error` (so a source with a lower
MAE recently counts for more; a source with no track record yet gets weight
1). Recomputed fresh every night in `service.blend_weighted`, stored in the
`ensemble_predictions` table.

### ML model

A separate Ridge Regression (continuous variables) or Logistic Regression
(`did_rain`) model **per location, per target variable**, trained on that
location's full history of every source's forecast plus cross-source
agreement features (mean/median/std/min/max/range/count across sources) and
date features (month, day of year, day of week). Stored in
`ml_predictions`. `precipitation_sum`, `wind_speed`, `wind_gusts`,
`cloud_cover`, `humidity`, and `pressure_msl` predictions are clipped at 0
before being stored or scored (`ml.clip_prediction`) — Ridge has no
non-negativity constraint and can otherwise predict a small negative value
for a quantity that physically can't go below zero. `cloud_cover` and
`humidity` are also capped at 100 (percentages can't physically exceed it) —
a missing-source data gap once let Ridge extrapolate a 467.9% prediction.

### Walk-forward backtest

`--backtest-days N` regenerates both predictions for each of the past N days
**as if that date were the present** — the weighted blend's MAE-weighting and
the ML model's training data only ever use rows strictly before the target
date, so nothing the model "shouldn't know yet" leaks in. A fresh model is
trained from scratch for every single date (this is what makes it a genuine
backtest rather than one model scored against its own future). Dates that
already have a real prediction are left untouched — it only fills gaps.

---

## Database

A single SQLite file, `data/weather.db` (committed to the repo so history
travels with the code):

| Table | One row per... |
|---|---|
| `forecasts` | source × location × forecast date × collection time |
| `actuals` | source × location × observed date |
| `ensemble_predictions` | location × forecast date × generation time (the weighted blend) |
| `ml_predictions` | location × forecast date × generation time (the ML model) |

`--dedupe` removes duplicate rows for the same source/location/date, keeping
**live over backfill** for `forecasts` (matching how every read query already
prioritizes them) and the **newest `generated_at`** everywhere else.

---

## The accuracy dashboard

`--accuracy-report PATH` builds a single self-contained interactive HTML file
(Plotly, loaded from a CDN):

- **Recent forecasts** — today's weighted/ML forecast plus the last several
  days' predictions alongside their actuals, for whichever location is
  selected in the dropdown (hidden until a location is picked — a single
  day's forecast isn't meaningful pooled across 30 locations).
- **Historical accuracy** — one card per weather variable: a leaderboard
  (mean absolute error, ranked best to worst) and a rolling MAE-over-time
  line chart, both covering every individual source, the weighted blend, the
  ML model, and two naive baselines (**persistence**: tomorrow = today;
  **30-day trailing average**). A "Show baselines" toggle removes them and
  rescales the chart to whatever's left. A location dropdown pools all 30
  locations by default or scopes everything to one.
- Every individual source gets its own shade along a fixed gold→maroon
  gradient (rather than one flat color) so they stay distinguishable without
  competing with the weighted blend (blue) and ML model (green), which are
  always drawn solid and on top.

The daily GitHub Actions workflow regenerates this report every night and (once
GitHub Pages is enabled for the repo — Settings → Pages → Source → "GitHub
Actions") publishes it to a stable URL that always reflects the latest run.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in whichever optional provider keys you
have — the pipeline works with zero keys using only Open-Meteo, wttr.in, and
BOM.

```powershell
weather-ensemble --help
pytest
```

## Quick start (single location)

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --deploy-phases 180
```

Backfills 180 days of actuals + historical forecasts, collects live forecasts,
builds the ML feature table, trains models, and predicts tomorrow.

## The real daily run (all 30 locations)

This is what the scheduled GitHub Action actually runs:

```powershell
weather-ensemble --all-locations --all --window 30 --train --train-window 90 --predict-ml
weather-ensemble --all-locations --accuracy-report reports/accuracy_report.html
```

---

## CLI reference

Global: `--db PATH` (default `data/weather.db`), `--all-locations` (repeat for
every configured location instead of one `--lat/--lon/--name/--timezone`).

| Flag | What it does |
|---|---|
| `--collect` | Collect tomorrow's forecast from every configured source |
| `--collect-open-meteo` | Collect only the free, no-key Open-Meteo models |
| `--record-actual` | Record yesterday's observed weather |
| `--backfill DAYS` | Backfill actuals + historical Open-Meteo forecasts for the past DAYS days |
| `--forecast` | Generate the weighted-average blend for tomorrow (`--window` sets the MAE-weighting lookback) |
| `--all` | Shorthand for collect + record-actual + forecast |
| `--train` | Train the ML models (`--train-window` sets the training lookback, default 90) |
| `--predict-ml` | Generate tomorrow's ML forecast |
| `--backtest-days DAYS` | Walk-forward re-create both predictions for each of the past DAYS days |
| `--dedupe` | Remove duplicate forecast/actual/prediction rows across the whole database |
| `--accuracy-report PATH` | Build the interactive HTML dashboard |
| `--report-window` / `--report-recent-days` / `--report-history-days` | Tune the dashboard's rolling-MAE smoothing window, leaderboard lookback, and total chart history (defaults 7 / 30 / 90 days) |
| `--deploy-phases DAYS` | One-shot: backfill + build feature table + train + predict, for a single location |
| `--export PATH` / `--build-dataset PATH` | Export the long modelling table / wide ML feature table to parquet or CSV |
| `--model-dir PATH` | Where trained model files live (default `models/`, one subfolder per location) |

---

## Repository structure

```text
weather-ensemble-forecaster/
├── src/weather_ensemble/
│   ├── cli.py           # command line interface
│   ├── config.py        # locations, variables, targets, env-driven settings
│   ├── db.py            # SQLite schema and insert/upsert functions
│   ├── models.py        # ForecastRecord / ActualRecord dataclasses
│   ├── service.py       # collection, backfill, MAE scoring, weighted blend
│   ├── ml.py             # feature building, training, ML prediction
│   ├── backtest.py      # walk-forward backtest of both predictions
│   ├── scoring.py        # long-format predictions-vs-actuals + rollups for the dashboard
│   ├── report.py         # the interactive HTML dashboard
│   ├── maintenance.py    # --dedupe
│   ├── retry.py          # shared HTTP retry-with-backoff
│   ├── phases.py         # one-command Phase 1-3 pipeline (single location)
│   └── sources/          # one fetcher module per provider (see table above)
├── tests/                # pytest suite
├── data/weather.db        # the SQLite database (committed - history travels with the code)
├── reports/accuracy_report.html  # the dashboard, regenerated nightly
├── models/                # trained ML model files, one subfolder per location (git-ignored)
├── .github/workflows/
│   ├── daily-collect.yml # the nightly pipeline + dashboard + Pages deploy
│   └── tests.yml         # lint + pytest on push/PR
├── notebooks/             # ad-hoc SQL/pandas exploration of data/weather.db
├── pyproject.toml
└── README.md
```
