# Weather Ensemble Forecaster

A pipeline that runs every night on GitHub Actions for 30 Australian locations. For each location it:

1. collects tomorrow's forecast from eleven weather providers;
2. records the weather that actually happened;
3. builds three predictions of its own: **Weighted**, **ML** and **Best**;
4. writes a short plain-English summary of the Best prediction with Claude;
5. scores every source and prediction against the observed weather.

Results are published as an interactive HTML accuracy dashboard on GitHub Pages.

Every forecast in this project is **one day ahead**: on day D the pipeline predicts day D+1, and that prediction is scored once D+1's observations exist.

---

## Contents

- [The nightly run](#the-nightly-run)
- [Per-location pipeline](#per-location-pipeline)
- [Forecast sources](#forecast-sources)
- [Actuals (ground truth)](#actuals-ground-truth)
- [Sub-daily rain periods](#sub-daily-rain-periods)
- [The three predictions](#the-three-predictions)
- [Weather narratives](#weather-narratives)
- [Walk-forward backtest](#walk-forward-backtest)
- [Scoring and baselines](#scoring-and-baselines)
- [The accuracy dashboard](#the-accuracy-dashboard)
- [Data storage](#data-storage)
- [Configuration](#configuration)
- [Setup and usage](#setup-and-usage)
- [CLI reference](#cli-reference)
- [Notebooks](#notebooks)
- [Repository structure](#repository-structure)

---

## The nightly run

Four GitHub Actions workflows (`.github/workflows/`) chain together:

| Workflow | Trigger | What it does |
|---|---|---|
| **Daily forecast collection** (`daily-collect.yml`) | cron `17 8 * * *` (08:17 UTC = 18:17 AEST / 19:17 AEDT), or manual | Runs the full pipeline for all 30 locations, then commits `data/weather.db`, `data/weather_periods.db` and `data/archive/` to `main`. |
| **Regenerate accuracy report** (`regenerate-report.yml`) | A successful daily collection, a push to `main` that changes either database, or manual | Rebuilds `reports/accuracy_report.html`, archives a snapshot of it in `data/weather_report_archive.db`, and commits both. |
| **Deploy accuracy report to Pages** (`deploy-pages.yml`) | A successful report regeneration, a push that changes the report, or manual | Publishes `reports/accuracy_report.html` as the GitHub Pages site's `index.html`. |
| **Tests** (`tests.yml`) | Every push and pull request | `ruff check src tests` and `pytest -q` on Python 3.11. |

GitHub often starts scheduled runs hours late, sometimes past local midnight in the eastern states. The pipeline is built for this: the date it forecasts comes from the data, not the wall clock (see [Target date](#target-date)).

The collection command:

```bash
weather-ensemble --all-locations \
  --all --window 30 --train --train-window 90 --predict-ml --predict-best --narrate-best \
  --collect-periods --record-actual-periods --forecast-periods \
  --train-periods --predict-ml-periods \
  --archive-forecasts-days 180
```

The report command:

```bash
weather-ensemble --all-locations --accuracy-report reports/accuracy_report.html
```

Both commit steps can collide with another workflow's push. To handle that, each step:

1. copies its output files aside;
2. hard-resets to the latest `origin/main`;
3. copies its files back, commits and pushes;
4. retries up to 5 times with a growing back-off, then fails the job.

Each workflow's own files always win, and no other files are touched. Concurrency groups stop two collection runs, or two report runs, from overlapping.

The collection workflow passes these secrets as environment variables: `WEATHERAPI_KEY`, `VISUAL_CROSSING_KEY`, `OPENWEATHERMAP_KEY`, `WEATHERBIT_KEY`, `ACCUWEATHER_KEY` and `ANTHROPIC_API_KEY`.

To publish the site, enable Pages under Settings → Pages → Source → "GitHub Actions".

---

## Per-location pipeline

`cli.main` first runs any whole-database maintenance it was asked for, in this order: `--dedupe`, `--export-blob-backlog`, `--archive-forecasts-days`. It then calls `cli._run_for_location` once per location. For the nightly flags, each location runs these steps in order:

| # | Step | Function |
|---|---|---|
| 1 | **Catch up missed dates.** Rebuild any recent date that has no Weighted prediction (see below). | `cli._catch_up_missed_forecasts` |
| 2 | **Collect** the target date's forecast from every source. | `service.collect_forecasts` |
| 3 | **Record actuals** for yesterday, plus any recent day still missing one. | `service.record_actual` |
| 4 | **Blend** the Weighted forecast. | `service.blend_forecast` |
| 5 | **Train** this location's ML models. | `ml.train_models` |
| 6 | **Predict ML.** | `ml.predict_latest_ml` |
| 7 | **Collect period forecasts** (sub-daily rain) from the Open-Meteo models. | `service.collect_forecast_periods` |
| 8 | **Record period actuals.** | `service.record_actual_periods` |
| 9 | **Blend period forecasts** for each of the 4 periods, then rescale them to sum to the daily Weighted total. | `service.blend_forecast_period`, `service.reconcile_period_predictions` |
| 10 | **Train period ML models**, one per period. | `ml.train_period_model` |
| 11 | **Predict period ML**, then rescale to the daily ML total. | `ml.predict_latest_ml_period`, `service.reconcile_period_predictions` |
| 12 | **Pick Best.** This runs after every Weighted, ML and period step, because it only copies values that already exist. | `best.predict_best` |
| 13 | **Narrate** the Best prediction. | `narrative.generate_and_store_best_narrative` |

<a id="target-date"></a>
### Target date

The date every forward-looking step works on (steps 2, 4, 6, 7, 9, 11, 12 and 13) is resolved **once per location per run** by `service.default_forecast_target_date`:

```
target = min(location's local tomorrow, latest ensemble_predictions date + 1)
```

A location with no ensemble history targets its local tomorrow. The value is computed lazily and cached. Each step receives it explicitly instead of resolving it again. Once step 4 writes a Weighted row, re-resolving would move every later step on to the following day.

A source that can't find that exact date in its response raises an error rather than substituting a neighbouring day.

### Catch-up of missed dates

`service.missing_ensemble_dates` looks back `CATCH_UP_LOOKBACK_DAYS` (10) days, up to yesterday, for dates with no `ensemble_predictions` row. A location with no ensemble history at all is treated as having no gaps. For any gap, step 1:

1. backfills the gap with `service.backfill`;
2. rebuilds Weighted and ML for those dates with `backtest.backtest_predictions`;
3. rebuilds Best with `backtest.backtest_best_predictions`;
4. writes a narrative for each missing date.

Backfill uses Open-Meteo's archived forecasts, which only cover completed days. A gap that is still in progress therefore gets picked up on the next run. When nothing is missing, catch-up is a single query. It doesn't rebuild period predictions.

### Catch-up of missing actuals

`service.actual_dates_to_record` drives both `record_actual` and `record_actual_periods`. It always re-fetches yesterday first, then any day in the same 10-day lookback with no actuals row. A location with no actuals yet records only yesterday.

### Failure isolation and exit code

- **Per source:** inside collection and backfill, each source and each Open-Meteo model runs in its own `try/except`. A failure prints `WARN: <source> failed: …`, with URL query strings (API keys) redacted, and collection continues.
- **Per step:** each per-location step runs inside `cli._guarded`. A failing step prints `WARN: <step> failed for <location>: …`, marks the location as failed, and the remaining steps and locations still run.
- **Single-location run:** exits 1 if any guarded step failed.
- **`--all-locations` run:** prints a summary of the failed locations, but exits 1 only if **every** location had a failure. That points to something systemic, such as an outage or expired keys, rather than one provider having a bad night.
- **Unguarded steps:** the reconcile calls and `--export`, `--build-dataset`, `--deploy-phases` and `--accuracy-report` are not wrapped, so an exception in any of them stops the run.

---

## Forecast sources

Each source is a `ForecastFetcher`, a function with the signature `fetch_forecast(location, target_date) -> ForecastRecord`, registered in `sources/__init__.py`'s `FORECAST_SOURCES`:

- It is always given the run's resolved target date.
- It matches the date in the provider's response by value, never by position.
- It raises `ValueError` if the date is missing.
- Keyed providers raise `RuntimeError` when their key isn't set.
- Every HTTP request goes through `retry.get_with_retry` with a 15-second timeout (see [Retries](#retries)).
- Live records are stored with `collection_method='live'` and `collected_at` in naive UTC.

| `source` | Provider and endpoint | Key | Backfill | Periods |
|---|---|---|---|---|
| `open_meteo_best_match` | Open-Meteo `/v1/forecast`, auto-selected "best" model | none | **No.** Open-Meteo's historical-forecast API returns the *observed* value for `best_match`, which would leak the answer into training. | Yes |
| `open_meteo_ecmwf_ifs025` | Open-Meteo, ECMWF IFS 0.25° | none | Yes | Yes |
| `open_meteo_gfs_global` | Open-Meteo, NOAA GFS Global | none | Yes | Yes |
| `open_meteo_gem_seamless` | Open-Meteo, Environment Canada GEM Seamless | none | Yes | Yes |
| `wttr_in` | `wttr.in/{lat},{lon}?format=j1` | none | No | No |
| `weatherapi` | WeatherAPI.com `/v1/forecast.json` (3 days) | `WEATHERAPI_KEY` | No | No |
| `visual_crossing` | Visual Crossing Timeline API, metric, days and hours | `VISUAL_CROSSING_KEY` | No | No |
| `openweathermap` | OpenWeatherMap `/data/2.5/forecast` (5-day / 3-hourly, free tier) | `OPENWEATHERMAP_KEY` | No | No |
| `weatherbit` | Weatherbit `/v2.0/forecast/daily` (3 days) | `WEATHERBIT_KEY` | No | No |
| `accuweather` | AccuWeather geoposition lookup, then `/forecasts/v1/daily/5day` with details | `ACCUWEATHER_KEY` | No | No |
| `bom` | Bureau of Meteorology's unofficial public JSON API (`api.weather.bom.gov.au`): location search, then daily forecast | none | No | No |

`bom` is reverse-engineered rather than an officially documented integration. Its responses carry a "must not use, copy or share" notice, so treat it as best-effort.

`visual_crossing` and `accuweather` are wired in, but the live database has no rows from either in the last 30 days, so their fetches are failing every night. Check those two keys first.

### Fields

Every source reports into the same ten fields: `max_temp`, `min_temp`, `rain_probability`, `precipitation_sum`, `wind_speed`, `wind_gusts`, `cloud_cover`, `humidity`, `pressure_msl` and `weather_code`. Units are °C, %, mm, km/h, % and hPa. A field a provider doesn't supply is stored as `NULL`, never estimated.

| Source | How it maps to the fields |
|---|---|
| Open-Meteo | Temperatures, `precipitation_probability_max`, wind and gusts come from the daily block. `precipitation_sum` is the sum of the hourly `precipitation`. Cloud, humidity and pressure are hourly means. `weather_code` is a WMO code. |
| `wttr_in` | `precipitation_sum` is always NULL: wttr.in gives a rain chance but no reliable daily total. Rain chance, wind and gusts are hourly maxima. Cloud, humidity and pressure are hourly means. |
| `weatherapi` | All fields. Gusts are the hourly maximum, and cloud and pressure are hourly means. |
| `visual_crossing` | All fields except `weather_code`, which is NULL because it reports conditions as text and icons. |
| `openweathermap` | Aggregated over the 3-hour blocks whose local date is the target: max temperature, min temperature, max rain chance, summed `rain.3h`, max wind and gust, mean cloud, humidity and pressure. Wind is converted from m/s to km/h. |
| `weatherbit` | All fields. Wind is converted from m/s to km/h. Pressure is sea-level `slp`, falling back to `pres`. |
| `accuweather` | All fields except `pressure_msl`. Values come from the forecast's "Day" part. |
| `bom` | Only temperatures, rain chance and `precipitation_sum`, which is the midpoint of BOM's min–max rain range. Everything else is NULL. |

`weather_code` is only a WMO code for Open-Meteo. The other providers use their own code systems: WWO, WeatherAPI's condition code, OpenWeatherMap's condition id, Weatherbit's code and AccuWeather's icon number. It is stored, but never blended or scored.

There is no UV index field. Every forecast API tested reported a clear-sky UV figure running 2–4× above observed UV, so `db.init_db` drops any `uv_index` column it finds.

### Configured but unused

These names are defined in `config.py` but nothing reads them:

- `OPTIONAL_OPEN_METEO_MODELS`: DWD ICON Global and BOM ACCESS Global via Open-Meteo.
- `OPTIONAL_LIVE_PROVIDERS`

<a id="retries"></a>
### Retries

`retry.get_with_retry` handles every HTTP request:

- **What it retries:** connection errors, timeouts, and HTTP 429, 500, 502, 503 and 504.
- **How often:** up to 4 times (5 attempts in all), sleeping 2, 4, 8 and 16 seconds between attempts.
- **What it doesn't retry:** any other 4xx, such as a bad request, invalid key or not found. That status is raised immediately, because trying again can't fix it.

---

## Actuals (ground truth)

**`open_meteo_archive`** is the single actuals source the pipeline uses. It comes from Open-Meteo's Historical Weather API (`archive-api.open-meteo.com`):

- **Daily:** `max_temp`, `min_temp`, `precipitation_sum`, `wind_speed` (daily max), `wind_gusts` (daily max) and `weather_code`.
- **Hourly means:** `cloud_cover`, `humidity` and `pressure_msl`.
- **`did_rain`:** `precipitation_sum >= RAIN_THRESHOLD_MM` (default 0.2 mm).
- **Sub-daily periods:** the hourly precipitation summed per period.

Actuals rows are upserted, so a later fetch for the same date replaces the earlier one.

**`silo`** (`sources/silo.py`) is the Queensland government's SILO gridded daily climate data, built from BOM's station network:

- It records max/min temperature, rain and `did_rain`, and needs only `SILO_EMAIL`.
- **The pipeline doesn't call it.** Every forecast-to-actual join (`service.load_modelling_table`) assumes exactly one actuals row per location and date, so a second actuals source would duplicate every forecast row in that join.
- Call `silo.fetch_actual` directly if you want an independent comparison.

---

## Sub-daily rain periods

Rain is also forecast and scored for four local-time periods, defined in `config.PERIOD_HOURS` with the end hour excluded:

| Period | Hours |
|---|---|
| `overnight` | 00–06 |
| `morning` | 06–12 |
| `afternoon` | 12–18 |
| `evening` | 18–24 |

- **Sources:** only the four Open-Meteo models report periods. A period's `precipitation_sum` is the sum of its hourly precipitation, and its `rain_probability` is the maximum hourly precipitation probability.
- **Storage:** all period data lives in its own database, `data/weather_periods.db` (see [Data storage](#data-storage)).
- **Weighted and ML:** each period is blended and modelled separately, for precipitation only (there is no per-period `did_rain`). `service.reconcile_period_predictions` then rescales the four periods so they sum to that prediction's daily total, keeping their shape. If every period is 0 but the daily total isn't, the total is split evenly. A date that is missing the daily value or any period is skipped.
- **Best:** copies the periods of whichever candidate won the daily rain pick, so Best's periods sum to its own daily total.
- **Dashboard:** periods appear as four extra targets, `precipitation_sum_<period>`.

---

## The three predictions

### Weighted (`ensemble`)

`service.blend_forecast` writes one row per location and date to `ensemble_predictions`. In the dashboard this series is labelled "Weighted".

**How the blend is weighted**

1. `service.load_modelling_table` reads the last `--window` days of forecasts joined to actuals and keeps only dates before the target date. The window defaults to `ROLLING_WINDOW_DAYS`, or 30. It takes one forecast per source and date, preferring live rows over backfilled ones, then the newest.
2. `service.compute_mae_scores` gives each source a mean absolute error (MAE) for each of eight variables: max/min temperature, precipitation, wind, gusts, cloud, humidity and pressure.

**How each value is formed**

- **Those eight variables:** `Σ(value × w) / Σw`, where `w = 1 / that source's MAE`. A source with no MAE yet, or an MAE of 0, gets weight 1. The result is rounded to 2 dp.
- **`rain_probability`:** a plain mean across sources.
- **`did_rain`:** blended `precipitation_sum >= RAIN_THRESHOLD_MM`.
- **No forecasts for the target date:** the step returns an error and writes nothing.

### ML (`ml`)

`ml.py` trains one model **per location and per target**. Each model is a scikit-learn pipeline: median imputation, then a learner.

| Target | Model |
|---|---|
| `did_rain` | `LogisticRegression(max_iter=1000)`, which also stores `did_rain_probability` |
| `max_temp`, `min_temp`, `precipitation_sum`, `wind_speed`, `wind_gusts`, `cloud_cover`, `humidity`, `pressure_msl` | `Ridge(alpha=1.0)` |

**Features.** `ml.features_for_target` uses only the target variable's own columns:

- every source's forecast of that variable (`{source}__{var}`);
- the cross-source mean, median, standard deviation, min, max and range;
- month, day of year and day of week.

`did_rain` uses the precipitation features.

If a source's value is missing on a row, it is filled with that row's cross-source median, after the spread statistics are computed. At prediction time a source with no column at all is filled the same way.

**Training.**

- The data is the last `--train-window` days (default 90) of forecasts joined to actuals.
- Training needs at least 30 rows.
- The rows are split chronologically into 75% train and 25% test, and the saved model is the one fitted on the train portion.
- The training summary reports MAE and RMSE for regression, and accuracy and AUC for `did_rain`.

**Saved files.** Models are saved to `models/<location-slug>/{target}.pkl`, with a `training_summary.json` beside them. `models/` is git-ignored, so the nightly run retrains from scratch every time.

**Clipping.** `ml.clip_prediction` keeps predictions physically possible, because Ridge has no bounds of its own:

| Targets | Bound |
|---|---|
| `precipitation_sum`, `wind_speed`, `wind_gusts`, `cloud_cover`, `humidity`, `pressure_msl` | floored at 0 |
| `cloud_cover`, `humidity` | capped at 100 |
| `precipitation_sum` | capped at 500 mm |
| temperatures | not clipped |

Precipitation is fitted with plain Ridge plus that ceiling. A log-transformed target scored markedly worse in backtests.

**Output.** Predictions go to `ml_predictions`.

**Period models.** One Ridge precipitation model per period, `precipitation_sum_{period}.pkl`, trained the same way. Its predictions go to `ml_predictions_periods`.

### Best (`best`)

Best is not a model. For each target, `best.py` copies the value of whichever **candidate** has had the lowest MAE **at that location** recently.

**Candidates.** Every raw source, plus Weighted and ML. The two baselines and Best itself are never candidates.

**Window.** A candidate is ranked on scored days in `[D − 30, D)` (`--best-window-days`, default 30), and needs at least 14 scored days in that window to be eligible (`--best-min-days`). `--best-window-days` is deliberately separate from the blend's `--window`.

**Fall-through.** Candidates are ranked by MAE. The first one that has a value for the target date wins, so a one-off collection gap at the top candidate falls through to the next-ranked one.

**Rain targets.** `precipitation_sum` and `did_rain` are picked together, by `best._select_precip_and_rain`:

- Only *period-aware* candidates are considered: the four Open-Meteo models, Weighted and ML.
- They are ranked on `precipitation_sum` MAE.
- The winner is the first that has both a daily value and period values for the date. If none has periods, the first with a daily value wins and Best's periods stay blank.
- `did_rain_probability` comes from the same winner. For raw sources and Weighted it is `rain_probability / 100`; for ML it is ML's own `did_rain_probability`.

**Periods.** Best's four period values are the rain winner's own, copied as they are. This keeps them consistent with Best's daily total.

Different targets at the same location routinely pick different winners, for example ECMWF for max temperature and Weighted for rain. The winner can change from day to day.

**Output.** Rows go to `best_predictions` and `best_predictions_periods`.

---

## Weather narratives

`narrative.py` (`--narrate-best`) makes one call to the Claude API per location, using the `anthropic` SDK with `claude-haiku-4-5` and `max_tokens=450`. It turns the Best prediction into a short paragraph.

**Input.** The prompt sends the location, weekday and date, along with:

- max/min temperature;
- rain chance, total rain and rain by period;
- cloud cover;
- wind speed and gusts.

**Rules in the system prompt (`SYSTEM_PROMPT`):**

- **Length:** at most 4–5 sentences.
- **Always covered:** temperature, rain (whether, how much and when), sun/cloud, and wind.
- **Tips:** an optional clothing tip and an optional activity tip, each with explicit rules.
- **Wording:** names the day rather than saying "today" or "tomorrow". Never mentions sources, models or probabilities.
- **Bands:** fixed tables translate numbers into words.

Each success inserts a row into `best_narratives`, and the dashboard shows the newest one. If there is no Best row, the result is `skipped: no_best_prediction`. If there is no API key, the `anthropic` package is missing, or the call fails, the result is `skipped: llm_unavailable`. A skip is never an error, so the narrative never fails the run.

`notebooks/narrative_prompt_playground.ipynb` is for trying prompt changes against real Best predictions without writing to the database.

---

## Walk-forward backtest

These commands rebuild predictions for past dates **as if each date were the present**. Every score, training row and ranking uses only data from before that date.

| Flag | What it rebuilds | Function |
|---|---|---|
| `--backtest-days N` | Weighted and ML | `backtest.backtest_predictions` |
| `--backtest-best-days N` | Best | `backtest.backtest_best_predictions` |
| `--backtest-periods-days N` | Weighted and ML periods, then reconciled to the daily totals | `backtest.backtest_period_predictions` |

- **Dates covered:** the last N dates that have actuals.
- **ML:** a fresh model is trained for every date, on the `--train-window` days before it (at least 30 rows, no holdout). These models are tagged as backtest models.
- **Gaps only:** a date that already has any row in the target table is skipped, so backtests only ever fill gaps. Rows are stamped `generated_at = D−1 21:00`, so re-runs are no-ops.
- **Reported counts:** written, skipped because a row already existed, and skipped because there was no forecast, too little data, or no eligible candidate.

The nightly catch-up step uses the daily and Best backtests to rebuild missed dates automatically.

---

## Scoring and baselines

`scoring.build_predictions_long` builds one long table with a row per model, location, date and target. Each row holds `predicted`, `actual`, `error` (predicted − actual) and `abs_error`. Only rows where both values are present are kept. The models are:

- every raw source;
- `ensemble`, `ml` and `best`;
- two baselines.

`scoring.build_period_predictions_long` adds the four `precipitation_sum_<period>` targets from the periods database.

**Targets.** `max_temp`, `min_temp`, `precipitation_sum`, `did_rain`, `wind_speed`, `wind_gusts`, `cloud_cover`, `humidity` and `pressure_msl`, plus the four period targets. `rain_probability` is not scored on its own.

**Rain (`did_rain`)** is scored as the absolute error between a 0–1 rain probability and the 0/1 outcome:

- raw sources and Weighted use `rain_probability / 100`;
- ML and Best use `did_rain_probability`.

**Baselines.** Both are computed from the actuals history alone, using only earlier dates:

| Baseline | How it predicts |
|---|---|
| **Persistence** (`baseline_persistence`) | Tomorrow will be the same as the previous observed day. For rain this is a hard 0/1. |
| **30-day trailing average** (`baseline_climatology`) | The mean of the previous 30 actuals rows, with at least 7 required. For rain this is the recent fraction of rainy days. |

Baselines show how much real forecasting beats doing nothing. They are never Best candidates.

**Summaries.** Two functions summarise the long table:

- `scoring.leaderboard` gives the MAE and count per target and model over the last `recent_days`.
- `scoring.rolling_error_over_time` gives a **centered** rolling mean of daily MAE. Errors are first averaged across locations per date, then each series is reindexed onto a full calendar. A missing date is a gap inside the window rather than being skipped, and still gets a smoothed value from the days around it.

---

## The accuracy dashboard

`--accuracy-report PATH` (`report.build_html_report`) writes one self-contained HTML file. Plotly 2.35.2 loads from its CDN, and all the data is embedded in the page.

**What it reads:**

- **History:** it reads `--report-history-days + --report-window` days of history. The extra `--report-window` days give the centered trend full context at the left edge.
- **Charts:** they cover the last `--report-history-days` days (default 90).
- **Locations:** with `--all-locations`, all 30 locations are included; otherwise only the single location.

**Header and filter bar.** The header shows how many locations are pooled, the date range and the generation time. A sticky bar holds:

- a **Location** dropdown: "All locations (pooled)", or one location;
- a **Show baselines** toggle, off by default;
- a **dark/light mode** button, which follows the system setting until clicked.

The page remembers these choices in `localStorage`. A `?location=<name>` URL parameter selects a location directly. Changing location re-scopes every chart in the browser without rebuilding anything.

**Recent forecasts.** This section appears only when a single location is selected, because one day's forecast means nothing pooled across 30 locations. It has tabs for the 5 most recent dates. Each tab shows:

- **An at-a-glance card for Best's prediction:**
  - a weather icon and condition label, with max/min temperature, rain chance, rain amount, wind and cloud;
  - the label comes from rain amount first: 20 mm or more is "Heavy rain", 5 mm or more is "Rain", 2 mm or more is "Showers";
  - otherwise the label comes from cloud cover: Sunny, Mostly sunny, Partly cloudy, Mostly cloudy or Overcast.
- **The Best narrative**, if one exists.
- **A table** with columns Weighted / ML / Best / Actual and rows for:
  - temperature, rain chance, precipitation (daily and per period);
  - wind, gusts, cloud, humidity and pressure.

**Historical accuracy.** There is one card per target, with a jump-nav and a legend. Each card holds:

- **Leaderboard:** a horizontal bar chart of MAE over the last `--report-recent-days` days (default 30), lowest first. It re-ranks for the selected location, and each model keeps its own colour wherever it lands. A collapsible "Table view" shows the pooled numbers.
- **Trend chart:** the centered `--report-window`-day (default 7) rolling MAE for every model. Raw sources are drawn underneath, then baselines, then Weighted, ML and Best on top. Pooled lines average all locations per date before smoothing.

**Colours.** Weighted is blue, ML green and Best purple, all drawn solid and on top. Persistence is a dashed grey line and the 30-day average a dotted grey line. Each raw source gets its own shade along a fixed gold→maroon gradient, so they stay distinguishable without competing with the three predictions. Both themes have tuned palettes.

**Data quality.** The final section, computed by `data_quality.compute_data_quality` over the last `--report-recent-days` days, has two headline figures:

- **Forecasts missing entirely:** the percentage of expected forecasts with no row. These are calendar days counted from each source's or model's first appearance at that location.
- **Fields missing in existing forecasts:** the percentage of expected fields that are NULL. Only fields the source has reported at least once count, so a field a provider never supplies isn't counted against it.

A breakdown per source and model follows the location dropdown.

**Snapshots.** Every report run saves a gzipped copy to `data/weather_report_archive.db` (see [Data storage](#data-storage)). Snapshots older than 30 days are pruned. Use `--list-report-snapshots` and `--extract-report-snapshot` to retrieve an older version.

---

## Data storage

Three SQLite files are committed to the repo, so the history travels with the code.

**`data/weather.db`** (main database; override with `WEATHER_DB_PATH`)

| Table | Unique on | Notes |
|---|---|---|
| `forecasts` | source, location, forecast date, `collected_at` | `collection_method` is `live` or `backfill` |
| `actuals` | source, location, actual date | upserted; source is `open_meteo_archive` |
| `ensemble_predictions` | location, forecast date, `generated_at` | Weighted |
| `ml_predictions` | location, forecast date, `generated_at` | includes `did_rain_probability` |
| `best_predictions` | location, forecast date, `generated_at` | records `window_days` and `min_days` |
| `best_narratives` | location, forecast date, `generated_at` | narrative text and model name |

**`data/weather_periods.db`** (sub-daily data; override with `WEATHER_PERIODS_DB_PATH`)

| Table | Unique on |
|---|---|
| `forecast_periods` | source, location, forecast date, period, `collected_at` |
| `actual_periods` | source, location, actual date, period |
| `ensemble_predictions_periods` | location, forecast date, period, `generated_at` |
| `ml_predictions_periods` | location, forecast date, period, `generated_at` |
| `best_predictions_periods` | location, forecast date, period, `generated_at` |

**`data/weather_report_archive.db`** (override with `WEATHER_REPORT_ARCHIVE_DB_PATH`)

- `report_snapshots`: one gzipped dashboard per report run, keyed by `generated_at`, with 30 days kept.

**Write and read rules**

- **Writes:** forecasts and predictions are insert-only, and each run adds rows with a new `collected_at` or `generated_at`. Actuals are upserts.
- **Reading forecasts:**
  - Scoring, training and blend weighting take one row per source, location and date, preferring `live` over `backfill`, then the newest.
  - The values blended for the target date itself are each source's newest row.
- **Reading predictions:** use the newest `generated_at`.
- **Schema setup:** `db.init_db` creates missing tables and columns, drops `uv_index`, and fills a missing `collection_method` from `raw_json`.

**`data/archive/`** holds Parquet files of data moved out of the live databases:

- `--archive-forecasts-days DAYS` moves `forecasts` and `forecast_periods` rows older than DAYS into `forecasts_history.parquet` and `forecast_periods_history.parquet`, de-duplicated on `id`, and deletes them from the databases. The nightly run uses 180 days.
- Actuals and every prediction table are kept forever.
- `--export-blob-backlog` moves the contents of the `raw_json` and `metadata_json` columns into `<table>_<column>.parquet` and clears them in the databases. It runs on demand, not nightly. Prediction tables write NULL `metadata_json`. Forecasts and actuals still write a small `raw_json` value (`{}`, or an endpoint tag on backfilled Open-Meteo rows), so a re-run archives whatever has been added since.

**Deduplication.** `--dedupe` (`maintenance.deduplicate`) removes duplicates across the main and periods databases and reports the counts before and after:

- forecasts keep `live` over `backfill`, then the newest `collected_at`;
- actuals keep the newest;
- predictions keep the newest `generated_at`;
- `best_narratives` and report snapshots aren't touched.

---

## Configuration

**Constants** (`config.py`):

- `AUSTRALIAN_LOCATIONS`: 30 locations with name, latitude, longitude and timezone. They are the capital cities, large regional centres (Gold Coast, Newcastle, Geelong, Cairns and others) and tourist spots (Lorne, Byron Bay, Broome, Yulara and others).
- `FORECAST_VARIABLES`: the ten forecast fields.
- `TARGETS`: the nine scored targets.
- `PERIOD_HOURS`: the four periods.
- `OPEN_METEO_MODELS`: the four live models.
- `OPEN_METEO_BACKFILL_MODELS`: the same list without `best_match`.

**Environment variables.** These are read from `.env` via python-dotenv. Copy `.env.example` to start.

| Variable | Default | Used for |
|---|---|---|
| `WEATHER_DB_PATH` | `data/weather.db` | Main database |
| `WEATHER_PERIODS_DB_PATH` | `<db stem>_periods.db` next to the main database | Periods database |
| `WEATHER_REPORT_ARCHIVE_DB_PATH` | `<db stem>_report_archive.db` next to the main database | Report snapshots |
| `DEFAULT_LOCATION_NAME` / `DEFAULT_LAT` / `DEFAULT_LON` / `DEFAULT_TIMEZONE` | Melbourne, -37.8136, 144.9631, Australia/Melbourne | The location used when no `--lat/--lon/--name/--timezone` and no `--all-locations` is given |
| `ROLLING_WINDOW_DAYS` | 30 | Default for `--window` |
| `RAIN_THRESHOLD_MM` | 0.2 | `did_rain` threshold |
| `WEATHERAPI_KEY`, `VISUAL_CROSSING_KEY`, `OPENWEATHERMAP_KEY`, `WEATHERBIT_KEY`, `ACCUWEATHER_KEY` | blank | Optional providers. A blank key makes that source fail with a warning and be skipped. |
| `ANTHROPIC_API_KEY` | blank | Narratives. If blank, narratives are skipped. |
| `SILO_EMAIL` | blank | Only for calling `silo.fetch_actual` directly |

---

## Setup and usage

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env   # then fill in whichever keys you have
weather-ensemble --help
pytest
```

The pipeline works with no keys at all, using the four Open-Meteo models, wttr.in and BOM.

**Bootstrap a single location.** The one-shot `--deploy-phases DAYS` command:

1. backfills DAYS days of actuals and archived Open-Meteo forecasts;
2. collects live forecasts for the target date and records actuals;
3. writes the ML feature table to `data/processed/features.parquet`;
4. trains with `min_rows = min(30, max(10, DAYS // 2))`;
5. predicts ML.

It ignores every other per-location flag.

```powershell
weather-ensemble --lat -37.8136 --lon 144.9631 --name Melbourne --timezone Australia/Melbourne --deploy-phases 180
```

**Run the nightly pipeline by hand.** Use the same commands as in [The nightly run](#the-nightly-run). `scripts/daily_run.sh` runs only `--all` for the default location (catch-up, collect, actuals and Weighted). It does no ML, Best, periods or report.

**Fill gaps in past predictions:**

```powershell
weather-ensemble --all-locations --backtest-days 30 --backtest-best-days 30 --backtest-periods-days 30
```

---

## CLI reference

The entry point is `weather-ensemble`, which runs `weather_ensemble.cli:main`. Running it with no action flag prints the help text.

**Location and database**

| Flag | Default | Meaning |
|---|---|---|
| `--lat`, `--lon`, `--name`, `--timezone` | `DEFAULT_*` env values | A single location |
| `--all-locations` | off | Repeat every per-location action for all 30 locations |
| `--db PATH` | `WEATHER_DB_PATH`, or `data/weather.db` | Main database; the periods and archive databases sit next to it |

**Daily pipeline**

| Flag | Meaning |
|---|---|
| `--collect` | Catch up missed dates, then collect the target date's forecast from every source |
| `--collect-open-meteo` | Collect from the four Open-Meteo models only (no catch-up) |
| `--record-actual` | Record yesterday's actuals, plus any recent day still missing |
| `--forecast` | Blend the Weighted forecast |
| `--window DAYS` | MAE lookback for the Weighted blend (default `ROLLING_WINDOW_DAYS`, or 30) |
| `--all` | `--collect --record-actual --forecast` |
| `--backfill DAYS` | Backfill actuals and period actuals, plus archived forecasts for the backfillable Open-Meteo models (daily and periods), for the past DAYS days |

**ML**

| Flag | Meaning |
|---|---|
| `--train` | Train the per-location ML models |
| `--train-window DAYS` | Training lookback (default 90) |
| `--predict-ml` | ML prediction for the target date |
| `--model-dir PATH` | Model directory (default `models/`, with one subfolder per location) |

**Best and narratives**

| Flag | Meaning |
|---|---|
| `--predict-best` | Best prediction for the target date |
| `--best-window-days DAYS` | Best's ranking lookback (default 30) |
| `--best-min-days DAYS` | Scored days needed to be eligible (default 14) |
| `--narrate-best` | Claude narrative of the Best prediction |

**Sub-daily periods**

| Flag | Meaning |
|---|---|
| `--collect-periods` | Collect Open-Meteo period forecasts |
| `--record-actual-periods` | Record period actuals |
| `--forecast-periods` | Blend each period, then reconcile to the daily Weighted total |
| `--train-periods` | Train the per-period precipitation models |
| `--predict-ml-periods` | Predict each period, then reconcile to the daily ML total |

**Backtests**

| Flag | Meaning |
|---|---|
| `--backtest-days DAYS` | Walk-forward Weighted and ML (gaps only) |
| `--backtest-best-days DAYS` | Walk-forward Best (gaps only) |
| `--backtest-periods-days DAYS` | Walk-forward period Weighted and ML, then reconcile |

**Dashboard**

| Flag | Meaning |
|---|---|
| `--accuracy-report PATH` | Build the dashboard and archive a snapshot |
| `--report-window DAYS` | Rolling-MAE window for the trend charts (default 7) |
| `--report-recent-days DAYS` | Lookback for the leaderboard and data-quality figures (default 30) |
| `--report-history-days DAYS` | How far back the charts go (default 90) |
| `--list-report-snapshots` | List archived snapshot timestamps, newest first |
| `--extract-report-snapshot TS` | Write that snapshot to `reports/archive/<TS>.html` |

**Data and maintenance**

| Flag | Meaning |
|---|---|
| `--export PATH` | Export the long forecast-vs-actual modelling table (`.csv`, otherwise Parquet) |
| `--build-dataset PATH` | Export the wide ML feature table (Parquet, falling back to CSV) |
| `--deploy-phases DAYS` | One-shot single-location bootstrap (see [Setup and usage](#setup-and-usage)) |
| `--dedupe` | Remove duplicate rows across both databases |
| `--archive-forecasts-days DAYS` | Move forecast rows older than DAYS to Parquet |
| `--export-blob-backlog` | Move `raw_json` / `metadata_json` contents to Parquet |

---

## Notebooks

None of the notebooks are part of the pipeline.

- `notebooks/explore_database.ipynb`: SQL and pandas exploration of `data/weather.db`, covering schema, row counts, latest forecasts, actuals, forecast-vs-actual joins and per-source MAE.
- `notebooks/ml_model_comparison.ipynb`: walk-forward comparison, for the default location, of simple baselines against linear, tree and hurdle models.
- `notebooks/narrative_prompt_playground.ipynb`: iterate on `narrative.SYSTEM_PROMPT` and the model settings against real Best predictions. It calls the Anthropic API and never writes to the database.

---

## Repository structure

```text
weather-ensemble-forecaster/
├── src/weather_ensemble/
│   ├── cli.py             # entry point: flag parsing, per-location step order, target date, catch-up
│   ├── config.py          # locations, variables, targets, periods, Open-Meteo models, env settings
│   ├── db.py              # SQLite schema, migrations, inserts and upserts for both databases
│   ├── models.py          # ForecastRecord / ActualRecord / period record dataclasses
│   ├── service.py         # collection, actuals, backfill, MAE weighting, Weighted blend, period reconcile
│   ├── ml.py              # feature tables, training, clipping, ML and period-ML prediction
│   ├── best.py            # Best candidate ranking and selection
│   ├── narrative.py       # Claude narrative of the Best prediction
│   ├── backtest.py        # walk-forward rebuild of Weighted, ML, Best and periods
│   ├── scoring.py         # long predictions-vs-actuals table, baselines, leaderboard, rolling MAE
│   ├── report.py          # the HTML dashboard
│   ├── data_quality.py    # missing-row and missing-field percentages for the dashboard
│   ├── report_archive.py  # gzipped report snapshots with 30-day retention
│   ├── archive.py         # moving old forecasts and JSON blobs to Parquet
│   ├── maintenance.py     # --dedupe
│   ├── retry.py           # HTTP retry with back-off
│   ├── phases.py          # --deploy-phases bootstrap
│   └── sources/           # one fetcher module per provider, plus silo.py (actuals)
├── tests/                 # pytest suite
├── notebooks/             # exploration and prompt playground
├── scripts/daily_run.sh   # minimal single-location --all run
├── data/
│   ├── weather.db                  # main database (committed)
│   ├── weather_periods.db          # sub-daily database (committed)
│   ├── weather_report_archive.db   # report snapshots (committed)
│   └── archive/                    # Parquet archives (committed)
├── reports/accuracy_report.html    # the dashboard, regenerated after every collection
├── models/                # trained models, one subfolder per location (git-ignored)
├── .github/workflows/     # daily-collect, regenerate-report, deploy-pages, tests
├── .env.example
└── pyproject.toml
```
