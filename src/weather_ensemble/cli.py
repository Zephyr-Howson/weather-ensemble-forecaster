from __future__ import annotations

import argparse
import json
import re
import sys
from functools import cache
from pathlib import Path

import pandas as pd

from weather_ensemble.archive import archive_old_forecasts, export_blob_backlog
from weather_ensemble.backtest import (
    backtest_best_predictions,
    backtest_period_predictions,
    backtest_predictions,
)
from weather_ensemble.best import DEFAULT_MIN_DAYS, DEFAULT_WINDOW_DAYS, predict_best
from weather_ensemble.config import (
    AUSTRALIAN_LOCATIONS,
    PERIODS,
    Location,
    get_db_path,
    get_default_location,
    get_rolling_window_days,
    local_today,
)
from weather_ensemble.maintenance import deduplicate
from weather_ensemble.ml import (
    build_feature_table,
    predict_latest_ml,
    predict_latest_ml_period,
    train_models,
    train_period_model,
)
from weather_ensemble.narrative import generate_and_store_best_narrative
from weather_ensemble.phases import deploy_all_phases
from weather_ensemble.report import build_html_report
from weather_ensemble.report_archive import (
    RETENTION_DAYS,
    list_report_snapshots,
    load_report_snapshot,
    save_report_snapshot,
)
from weather_ensemble.scoring import build_period_predictions_long, build_predictions_long
from weather_ensemble.service import (
    _safe_error,
    backfill,
    blend_forecast,
    blend_forecast_period,
    collect_forecast_periods,
    collect_forecasts,
    collect_open_meteo_only,
    default_forecast_target_date,
    export_modelling_table,
    missing_ensemble_dates,
    reconcile_period_predictions,
    record_actual,
    record_actual_periods,
)


def _location_from_args(args: argparse.Namespace) -> Location:
    default = get_default_location()
    return Location(
        name=args.name or default.name,
        lat=args.lat if args.lat is not None else default.lat,
        lon=args.lon if args.lon is not None else default.lon,
        timezone=args.timezone or default.timezone,
    )


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _location_model_dir(base: Path, location: Location) -> Path:
    """Give each location its own model subdirectory.

    Trained models are one-per-location (fit on that location's own history), so
    sharing a single directory across --all-locations would let each location's
    training overwrite the previous one's .pkl files.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", location.name.lower()).strip("_") or "location"
    return base / slug


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weather ensemble forecaster")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--name", help="Location name")
    parser.add_argument("--timezone", help="Timezone, e.g. Australia/Melbourne")
    parser.add_argument("--db", type=Path, default=get_db_path(), help="SQLite DB path")
    parser.add_argument(
        "--window",
        type=int,
        default=get_rolling_window_days(),
        help="Lookback (days) of scored history used to MAE-weight each source in the Weighted blend "
        "(default ROLLING_WINDOW_DAYS env var, else 30)",
    )

    parser.add_argument(
        "--collect",
        action="store_true",
        help="Collect every source's forecast for tomorrow (first catching up any recently "
        "missed dates)",
    )
    parser.add_argument("--collect-open-meteo", action="store_true", help="Collect only free Open-Meteo model forecasts")
    parser.add_argument("--record-actual", action="store_true", help="Record yesterday's actual weather (plus any recent day still missing one)")
    parser.add_argument(
        "--backfill", type=int, metavar="DAYS", help="Backfill forecasts and actuals (daily and sub-daily rain periods)"
    )
    parser.add_argument("--forecast", action="store_true", help="Generate weighted-average blended forecast")
    parser.add_argument("--all", action="store_true", help="Run collect, record actual, and weighted forecast")
    parser.add_argument("--export", type=Path, help="Export long modelling table to parquet/csv")

    parser.add_argument(
        "--build-dataset",
        type=Path,
        help="Build wide ML feature table and save to parquet/csv",
    )
    parser.add_argument("--train", action="store_true", help="Train the per-location ML models")
    parser.add_argument(
        "--train-window",
        type=int,
        default=90,
        help="Days of history for ML model training (default 90; separate from the blend's --window)",
    )
    parser.add_argument("--predict-ml", action="store_true", help="Predict using trained ML models")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory for trained model files",
    )
    parser.add_argument(
        "--deploy-phases",
        type=int,
        metavar="DAYS",
        help="One-shot setup for a single location: backfill DAYS, collect, record actuals, build the "
        "feature table, train and predict ML. Ignores every other per-location flag",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        metavar="DAYS",
        help="Walk-forward: regenerate ensemble+ML predictions for each of the past DAYS days, "
        "using only data available before that date. Skips dates that already have a prediction.",
    )
    parser.add_argument(
        "--predict-best",
        action="store_true",
        help="Generate tomorrow's adaptive 'Best' prediction: per target, copies whichever candidate "
        "(a raw source, Weighted, or ML) has had the lowest MAE in this location over the trailing "
        "--best-window-days days (needs --best-min-days scored days to be eligible)",
    )
    parser.add_argument(
        "--best-window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"Trailing lookback (days) Best ranks candidates over - separate from --window, the "
        f"Weighted blend's MAE lookback (default {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--best-min-days",
        type=int,
        default=DEFAULT_MIN_DAYS,
        help=f"Minimum scored days within --best-window-days required for a candidate to be eligible "
        f"to win 'Best' for a target (default {DEFAULT_MIN_DAYS})",
    )
    parser.add_argument(
        "--backtest-best-days",
        type=int,
        metavar="DAYS",
        help="Walk-forward: regenerate the adaptive 'Best' prediction for each of the past DAYS days, "
        "using only data available before that date. Skips dates that already have a prediction.",
    )
    parser.add_argument(
        "--narrate-best",
        action="store_true",
        help="Generate a brief natural-language weather narrative for tomorrow's 'Best' prediction via "
        "a single Claude API call (requires ANTHROPIC_API_KEY - silently skipped, not an error, if unset "
        "or the call fails for any reason). Needs --predict-best to have already run for this date.",
    )
    parser.add_argument(
        "--all-locations",
        action="store_true",
        help="Repeat the requested actions for every location in AUSTRALIAN_LOCATIONS "
        "instead of a single --lat/--lon location",
    )

    parser.add_argument(
        "--accuracy-report",
        type=Path,
        metavar="PATH",
        help="Write an interactive HTML report scoring Weighted/ML/Best/individual sources/baselines "
        "against actuals over time (pools every location when combined with --all-locations)",
    )
    parser.add_argument(
        "--report-window",
        type=int,
        default=7,
        help="Rolling window (days) for the accuracy-over-time chart (default 7)",
    )
    parser.add_argument(
        "--report-recent-days",
        type=int,
        default=30,
        help="Lookback window (days) for the leaderboard summary (default 30)",
    )
    parser.add_argument(
        "--report-history-days",
        type=int,
        default=90,
        help="How far back every chart in the accuracy report goes (default 90, ~3 months)",
    )
    parser.add_argument(
        "--list-report-snapshots",
        action="store_true",
        help=f"List archived report snapshots (generated_at timestamps, newest first) - see "
        f"report_archive.py. Snapshots older than {RETENTION_DAYS} days are pruned automatically "
        f"whenever --accuracy-report saves a new one.",
    )
    parser.add_argument(
        "--extract-report-snapshot",
        metavar="GENERATED_AT",
        help="Write one archived snapshot's HTML to reports/archive/<GENERATED_AT>.html - "
        "see --list-report-snapshots for available timestamps",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Remove duplicate rows across every prediction/observation table in both databases "
        "(same source/location/date[/period]), keeping the highest-priority one (whole database, not per-location)",
    )
    parser.add_argument(
        "--export-blob-backlog",
        action="store_true",
        help="Move every write-only JSON blob column's existing content (forecasts/actuals "
        "raw_json, every prediction table's metadata_json) to Parquet under data/archive/, then "
        "clear it from the live row. Safe to re-run.",
    )
    parser.add_argument(
        "--archive-forecasts-days",
        type=int,
        metavar="DAYS",
        help="Move forecasts/forecast_periods rows older than DAYS to Parquet under data/archive/, "
        "then delete them from the live database. Actuals and every prediction table are kept "
        "forever and untouched by this. Safe to run on a schedule.",
    )

    # Sub-daily (overnight/morning/afternoon/evening) rain prediction. These are
    # separate opt-in flags, not part of --all or --deploy-phases. Open-Meteo
    # models only, and precipitation_sum only (no did_rain classifier).
    parser.add_argument(
        "--collect-periods", action="store_true", help="Collect sub-daily rain forecasts for the target date"
    )
    parser.add_argument(
        "--record-actual-periods", action="store_true", help="Record yesterday's sub-daily rain actuals (plus any recent gap)"
    )
    parser.add_argument(
        "--forecast-periods",
        action="store_true",
        help="Generate weighted-average sub-daily rain forecast for every period",
    )
    parser.add_argument(
        "--train-periods", action="store_true", help="Train per-period precipitation models"
    )
    parser.add_argument(
        "--predict-ml-periods",
        action="store_true",
        help="Predict sub-daily rain using trained per-period models",
    )
    parser.add_argument(
        "--backtest-periods-days",
        type=int,
        metavar="DAYS",
        help="Walk-forward: regenerate sub-daily rain ensemble+ML predictions for each "
        "of the past DAYS days. Needed for the report's ensemble/ML leaderboard entries to have "
        "anything to show - a live prediction has no actual yet to score against.",
    )
    return parser


def _write_dataframe(df, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
        return path
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception:  # noqa: BLE001 - parquet can fail for many library/serialization
        # reasons (missing pyarrow, an unsupported dtype); any of them should
        # fall back to CSV rather than crash the export.
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _guarded(location: Location, step: str, fn) -> bool:
    """Run one location's step in isolation from every other step and location.

    Any exception is logged as a WARN line (with API keys redacted) and reported
    as False, so one transient failure (a timeout, an upstream 5xx) for one
    location doesn't stop the remaining steps and locations from running. This
    is the per-step counterpart of collect_forecasts's per-source try/except.
    """
    try:
        fn()
        return True
    except Exception as exc:  # noqa: BLE001 - deliberately catch-all, see docstring
        print(f"WARN: {step} failed for {location.name}: {_safe_error(exc)}")
        return False


def _catch_up_missed_forecasts(args: argparse.Namespace, location: Location) -> dict:
    """Backfill and walk-forward-reconstruct any recent date this location has
    no ensemble prediction for, before today's normal collection runs.

    service.missing_ensemble_dates finds dates before local today with no
    ensemble_predictions row (within CATCH_UP_LOOKBACK_DAYS). For a gap
    starting N days ago this backfills N days of actuals and Open-Meteo
    historical forecasts, daily and per period (Open-Meteo is the only source
    with an archived-forecast endpoint), walk-forward reconstructs Weighted/ML
    and then Best for the latest N days with data (as --backtest-days/
    --backtest-best-days do, skipping dates that already have a prediction),
    and generates a narrative for each missing date. Only completed days are
    considered, because Open-Meteo's historical-forecast archive only covers
    those; a date still in progress is picked up by a later run. Costs a
    couple of SELECTs when nothing is missing.
    """
    missing = missing_ensemble_dates(args.db, location)
    if not missing:
        return {"missing_dates": []}

    gap_days = (local_today(location) - missing[0]).days
    backfill(args.db, location, gap_days)
    backtest_predictions(args.db, location, days=gap_days, ensemble_window_days=args.window, train_window_days=args.train_window)
    backtest_best_predictions(args.db, location, days=gap_days, window_days=args.best_window_days, min_days=args.best_min_days)

    narrated = [
        target_date.isoformat()
        for target_date in missing
        if "narrative" in generate_and_store_best_narrative(args.db, location, target_date=target_date)
    ]
    return {"missing_dates": [d.isoformat() for d in missing], "gap_days": gap_days, "narrated": narrated}


def _run_for_location(args: argparse.Namespace, location: Location) -> bool:
    model_dir = _location_model_dir(args.model_dir, location)
    ok = True

    if args.deploy_phases:
        result = deploy_all_phases(
            db_path=args.db,
            location=location,
            days_back=args.deploy_phases,
            processed_path=Path("data/processed/features.parquet"),
            model_dir=model_dir,
            train_window_days=args.train_window,
        )
        _print_json(result)
        return ok

    if args.collect or args.all:
        def _catch_up():
            result = _catch_up_missed_forecasts(args, location)
            _print_json(result)
        ok &= _guarded(location, "catch_up_missed_forecasts", _catch_up)

    # The date every forward-looking step below works on, resolved once per
    # location per run and passed to each step explicitly. It is resolved
    # lazily (only if one of those steps runs), which is always after the
    # catch-up above and before --forecast writes anything.
    # default_forecast_target_date is "one past the latest ensemble row, capped
    # at local tomorrow", so re-resolving it after --forecast has written that
    # row would move later steps on to the following day whenever the run
    # starts after local midnight (GitHub's scheduled runs can start hours late).
    target_date = cache(lambda: default_forecast_target_date(args.db, location))

    if args.backfill:
        def _backfill():
            backfill(args.db, location, args.backfill)
            print(f"Backfilled {args.backfill} days for {location.name} into {args.db}")
        ok &= _guarded(location, "backfill", _backfill)

    if args.collect_open_meteo:
        def _collect_open_meteo():
            records = collect_open_meteo_only(args.db, location, target_date=target_date())
            print(f"Collected {len(records)} Open-Meteo model forecast records")
        ok &= _guarded(location, "collect_open_meteo", _collect_open_meteo)

    if args.collect or args.all:
        def _collect():
            records = collect_forecasts(args.db, location, target_date=target_date())
            print(f"Collected {len(records)} forecast records")
        ok &= _guarded(location, "collect", _collect)

    if args.record_actual or args.all:
        def _record_actual():
            dates = record_actual(args.db, location)
            print(f"Recorded actual weather for {', '.join(d.isoformat() for d in dates)}")
        ok &= _guarded(location, "record_actual", _record_actual)

    if args.forecast or args.all:
        def _forecast():
            result = blend_forecast(args.db, location, args.window, target_date=target_date())
            _print_json(result)
        ok &= _guarded(location, "blend_forecast", _forecast)

    if args.export:
        output = export_modelling_table(args.db, location, args.export)
        print(f"Exported long modelling table to {output}")

    if args.build_dataset:
        df = build_feature_table(args.db, location)
        output = _write_dataframe(df, args.build_dataset)
        print(f"Exported wide ML feature table to {output} ({len(df)} rows, {len(df.columns)} columns)")

    if args.train:
        def _train():
            result = train_models(args.db, location, model_dir, window_days=args.train_window)
            _print_json(result)
        ok &= _guarded(location, "train", _train)

    if args.predict_ml:
        def _predict_ml():
            result = predict_latest_ml(args.db, location, model_dir, target_date=target_date())
            _print_json(result)
        ok &= _guarded(location, "predict_ml", _predict_ml)

    if args.backtest_days:
        def _backtest():
            result = backtest_predictions(
                args.db, location, days=args.backtest_days, ensemble_window_days=args.window, train_window_days=args.train_window
            )
            _print_json(result)
        ok &= _guarded(location, "backtest", _backtest)

    if args.collect_periods:
        def _collect_periods():
            n = collect_forecast_periods(args.db, location, target_date=target_date())
            print(f"Collected {n} sub-daily rain forecast rows")
        ok &= _guarded(location, "collect_periods", _collect_periods)

    if args.record_actual_periods:
        def _record_actual_periods():
            dates = record_actual_periods(args.db, location)
            print(f"Recorded sub-daily rain actuals for {', '.join(d.isoformat() for d in dates)}")
        ok &= _guarded(location, "record_actual_periods", _record_actual_periods)

    if args.forecast_periods:
        forecast_dates: set[str] = set()
        for period in PERIODS:
            def _forecast_period(period=period):
                result = blend_forecast_period(args.db, location, period, args.window, target_date=target_date())
                _print_json(result)
                if "forecast_date" in result:
                    forecast_dates.add(result["forecast_date"])
            ok &= _guarded(location, f"blend_forecast_period[{period}]", _forecast_period)
        if forecast_dates:
            recon = reconcile_period_predictions(args.db, location, "ensemble", sorted(forecast_dates))
            print(f"Reconciled ensemble periods to daily total: {recon}")

    if args.train_periods:
        for period in PERIODS:
            def _train_period(period=period):
                result = train_period_model(args.db, location, period, model_dir, window_days=args.train_window)
                _print_json(result)
            ok &= _guarded(location, f"train_period[{period}]", _train_period)

    if args.predict_ml_periods:
        forecast_dates = set()
        for period in PERIODS:
            def _predict_ml_period(period=period):
                result = predict_latest_ml_period(args.db, location, period, model_dir, target_date=target_date())
                _print_json(result)
                if "forecast_date" in result:
                    forecast_dates.add(result["forecast_date"])
            ok &= _guarded(location, f"predict_ml_period[{period}]", _predict_ml_period)
        if forecast_dates:
            recon = reconcile_period_predictions(args.db, location, "ml", sorted(forecast_dates))
            print(f"Reconciled ML periods to daily total: {recon}")

    if args.predict_best:
        # Runs after --forecast/--predict-ml/--collect-periods/
        # --forecast-periods/--predict-ml-periods above, because Best only
        # copies an already-computed candidate's value: this run's Weighted/ML
        # daily predictions must already exist, and so must this run's period
        # rows (reconciled to their daily totals), since precipitation_sum/
        # did_rain are chosen from PERIOD_AWARE_CANDIDATES and Best copies the
        # winner's period breakdown (see best.py).
        def _predict_best():
            result = predict_best(
                args.db, location, window_days=args.best_window_days, min_days=args.best_min_days, target_date=target_date()
            )
            _print_json(result)
        ok &= _guarded(location, "predict_best", _predict_best)

    if args.narrate_best:
        # Runs after --predict-best above: it reads whatever Best picked for
        # the target date (periods included) and asks Claude for a short
        # plain-English summary. See narrative.py: any failure (no API key,
        # network error, rate limit, malformed reply) comes back as a
        # "skipped" result, not an exception.
        def _narrate_best():
            result = generate_and_store_best_narrative(args.db, location, target_date=target_date())
            _print_json(result)
        ok &= _guarded(location, "narrate_best", _narrate_best)

    if args.backtest_best_days:
        def _backtest_best():
            result = backtest_best_predictions(
                args.db, location, days=args.backtest_best_days, window_days=args.best_window_days, min_days=args.best_min_days
            )
            _print_json(result)
        ok &= _guarded(location, "backtest_best", _backtest_best)

    if args.backtest_periods_days:
        date_ranges = []
        for period in PERIODS:
            def _backtest_period(period=period):
                result = backtest_period_predictions(
                    args.db, location, period, days=args.backtest_periods_days,
                    ensemble_window_days=args.window, train_window_days=args.train_window,
                )
                _print_json(result)
                if "date_range" in result:
                    date_ranges.append(result["date_range"])
            ok &= _guarded(location, f"backtest_period[{period}]", _backtest_period)
        if date_ranges:
            start = min(r[0] for r in date_ranges)
            end = max(r[1] for r in date_ranges)
            all_dates = [d.date().isoformat() for d in pd.date_range(start, end)]
            recon_ens = reconcile_period_predictions(args.db, location, "ensemble", all_dates)
            recon_ml = reconcile_period_predictions(args.db, location, "ml", all_dates)
            print(f"Reconciled historical ensemble periods: {recon_ens}")
            print(f"Reconciled historical ML periods: {recon_ml}")

    return ok


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    per_location_actions = [
        args.backfill,
        args.collect,
        args.collect_open_meteo,
        args.record_actual,
        args.forecast,
        args.all,
        args.export,
        args.build_dataset,
        args.train,
        args.predict_ml,
        args.deploy_phases,
        args.backtest_days,
        args.predict_best,
        args.narrate_best,
        args.backtest_best_days,
        args.collect_periods,
        args.record_actual_periods,
        args.forecast_periods,
        args.train_periods,
        args.predict_ml_periods,
        args.backtest_periods_days,
    ]

    if (
        not any(per_location_actions)
        and not args.accuracy_report
        and not args.dedupe
        and not args.export_blob_backlog
        and args.archive_forecasts_days is None
        and not args.list_report_snapshots
        and not args.extract_report_snapshot
    ):
        parser.print_help()
        return

    if args.dedupe:
        result = deduplicate(args.db)
        _print_json(result)

    if args.export_blob_backlog:
        result = export_blob_backlog(args.db)
        _print_json(result)

    if args.archive_forecasts_days is not None:
        result = archive_old_forecasts(args.db, cutoff_days=args.archive_forecasts_days)
        _print_json(result)

    exit_code = 0
    if any(per_location_actions):
        if args.all_locations:
            failed_locations = []
            for location in AUSTRALIAN_LOCATIONS:
                print(f"=== {location.name} ===")
                if not _run_for_location(args, location):
                    failed_locations.append(location.name)
            if failed_locations:
                print(
                    f"WARN: {len(failed_locations)}/{len(AUSTRALIAN_LOCATIONS)} location(s) "
                    f"had at least one failed step: {', '.join(failed_locations)}"
                )
                # A few locations hitting a transient error shouldn't fail the
                # run (the other locations' data is still good and still gets
                # committed), but every location failing points at something
                # systemic (bad credentials, an outage), so that exits 1.
                if len(failed_locations) == len(AUSTRALIAN_LOCATIONS):
                    exit_code = 1
        else:
            if not _run_for_location(args, _location_from_args(args)):
                exit_code = 1

    if args.accuracy_report:
        locations = AUSTRALIAN_LOCATIONS if args.all_locations else [_location_from_args(args)]
        # build_html_report only uses the most recent report_history_days;
        # +report_window is a buffer so the centered rolling-MAE trend has full
        # context at the edge of that window too.
        window_days = args.report_history_days + args.report_window
        long_df = build_predictions_long(args.db, locations, window_days=window_days)
        # Sub-daily rain periods are concatenated in as extra synthetic targets
        # (see build_period_predictions_long) rather than given a separate
        # rendering path.
        period_long_df = build_period_predictions_long(args.db, locations, window_days=window_days)
        if not period_long_df.empty:
            long_df = pd.concat([long_df, period_long_df], ignore_index=True) if not long_df.empty else period_long_df
        output = build_html_report(
            long_df,
            args.accuracy_report,
            args.db,
            rolling_window=args.report_window,
            recent_days=args.report_recent_days,
            history_days=args.report_history_days,
        )
        print(f"Wrote accuracy report to {output} ({len(long_df)} scored rows across {len(locations)} location(s))")

        snapshot = save_report_snapshot(args.db, output.read_text(encoding="utf-8"))
        print(f"Archived report snapshot: {snapshot}")

    if args.list_report_snapshots:
        for generated_at in list_report_snapshots(args.db):
            print(generated_at)

    if args.extract_report_snapshot:
        html = load_report_snapshot(args.db, args.extract_report_snapshot)
        if html is None:
            print(f"No archived snapshot found for {args.extract_report_snapshot!r} (see --list-report-snapshots)")
        else:
            safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", args.extract_report_snapshot)
            extract_path = Path("reports/archive") / f"{safe_name}.html"
            extract_path.parent.mkdir(parents=True, exist_ok=True)
            extract_path.write_text(html, encoding="utf-8")
            print(f"Extracted archived report to {extract_path}")

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
