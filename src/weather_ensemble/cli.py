from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from weather_ensemble.backtest import backtest_period_predictions, backtest_predictions
from weather_ensemble.config import (
    AUSTRALIAN_LOCATIONS,
    PERIODS,
    Location,
    get_db_path,
    get_default_location,
    get_rolling_window_days,
)
from weather_ensemble.maintenance import deduplicate
from weather_ensemble.ml import (
    build_feature_table,
    predict_latest_ml,
    predict_latest_ml_period,
    train_models,
    train_period_model,
)
from weather_ensemble.phases import deploy_all_phases
from weather_ensemble.report import build_html_report
from weather_ensemble.scoring import build_period_predictions_long, build_predictions_long
from weather_ensemble.service import (
    _safe_error,
    backfill,
    blend_forecast,
    blend_forecast_period,
    collect_forecast_periods,
    collect_forecasts,
    collect_open_meteo_only,
    export_modelling_table,
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
    parser.add_argument("--window", type=int, default=get_rolling_window_days())

    parser.add_argument("--collect", action="store_true", help="Collect forecasts for tomorrow")
    parser.add_argument("--collect-open-meteo", action="store_true", help="Collect only free Open-Meteo model forecasts")
    parser.add_argument("--record-actual", action="store_true", help="Record yesterday's actual weather")
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
    parser.add_argument("--train", action="store_true", help="Train Phase 3 ML models")
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
        help="Run Phase 1 backfill, Phase 2 dataset build, and Phase 3 model training",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        metavar="DAYS",
        help="Walk-forward: regenerate ensemble+ML predictions for each of the past DAYS days, "
        "using only data available before that date. Skips dates that already have a prediction.",
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
        help="Write an interactive HTML report scoring ensemble/ML/individual sources/baselines "
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
        "--dedupe",
        action="store_true",
        help="Remove duplicate rows across forecasts/actuals/ensemble_predictions/ml_predictions "
        "(same source/location/date), keeping the highest-priority one (whole database, not per-location)",
    )

    # Small-slice sub-daily (morning/afternoon/evening) rain prediction - kept
    # as separate, explicit, opt-in flags rather than folded into --all/
    # --deploy-phases, so nothing about the existing daily pipeline's
    # behavior changes until this is validated. Open-Meteo only (see
    # ForecastPeriodRecord); precipitation_sum only, no did_rain classifier.
    parser.add_argument(
        "--collect-periods", action="store_true", help="[small slice] Collect sub-daily rain forecasts for tomorrow"
    )
    parser.add_argument(
        "--record-actual-periods", action="store_true", help="[small slice] Record yesterday's sub-daily rain actuals"
    )
    parser.add_argument(
        "--forecast-periods",
        action="store_true",
        help="[small slice] Generate weighted-average sub-daily rain forecast for every period",
    )
    parser.add_argument(
        "--train-periods", action="store_true", help="[small slice] Train per-period precipitation models"
    )
    parser.add_argument(
        "--predict-ml-periods",
        action="store_true",
        help="[small slice] Predict sub-daily rain using trained per-period models",
    )
    parser.add_argument(
        "--backtest-periods-days",
        type=int,
        metavar="DAYS",
        help="[small slice] Walk-forward: regenerate sub-daily rain ensemble+ML predictions for each "
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

    Mirrors the per-source try/except collect_forecasts already uses (one bad
    provider shouldn't sink the whole collection run) - but the per-location
    steps here (record_actual, blend_forecast, train, predict_ml, backtest)
    previously had no such guard, so a single transient failure (a timeout, an
    upstream 5xx) for any one of the 30 locations would crash the whole
    --all-locations run and skip every step - and location - after it,
    including the report regeneration and the daily commit. A step failing
    here is logged and the run moves on instead.
    """
    try:
        fn()
        return True
    except Exception as exc:  # noqa: BLE001 - deliberately catch-all, see docstring
        print(f"WARN: {step} failed for {location.name}: {_safe_error(exc)}")
        return False


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

    if args.backfill:
        def _backfill():
            backfill(args.db, location, args.backfill)
            print(f"Backfilled {args.backfill} days for {location.name} into {args.db}")
        ok &= _guarded(location, "backfill", _backfill)

    if args.collect_open_meteo:
        def _collect_open_meteo():
            records = collect_open_meteo_only(args.db, location)
            print(f"Collected {len(records)} Open-Meteo model forecast records")
        ok &= _guarded(location, "collect_open_meteo", _collect_open_meteo)

    if args.collect or args.all:
        def _collect():
            records = collect_forecasts(args.db, location)
            print(f"Collected {len(records)} forecast records")
        ok &= _guarded(location, "collect", _collect)

    if args.record_actual or args.all:
        def _record_actual():
            record_actual(args.db, location)
            print("Recorded yesterday's actual weather")
        ok &= _guarded(location, "record_actual", _record_actual)

    if args.forecast or args.all:
        def _forecast():
            result = blend_forecast(args.db, location, args.window)
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
            result = predict_latest_ml(args.db, location, model_dir)
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
            n = collect_forecast_periods(args.db, location)
            print(f"Collected {n} sub-daily rain forecast rows")
        ok &= _guarded(location, "collect_periods", _collect_periods)

    if args.record_actual_periods:
        def _record_actual_periods():
            record_actual_periods(args.db, location)
            print("Recorded yesterday's sub-daily rain actuals")
        ok &= _guarded(location, "record_actual_periods", _record_actual_periods)

    if args.forecast_periods:
        forecast_dates: set[str] = set()
        for period in PERIODS:
            def _forecast_period(period=period):
                result = blend_forecast_period(args.db, location, period, args.window)
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
                result = predict_latest_ml_period(args.db, location, period, model_dir)
                _print_json(result)
                if "forecast_date" in result:
                    forecast_dates.add(result["forecast_date"])
            ok &= _guarded(location, f"predict_ml_period[{period}]", _predict_ml_period)
        if forecast_dates:
            recon = reconcile_period_predictions(args.db, location, "ml", sorted(forecast_dates))
            print(f"Reconciled ML periods to daily total: {recon}")

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
        args.collect_periods,
        args.record_actual_periods,
        args.forecast_periods,
        args.train_periods,
        args.predict_ml_periods,
        args.backtest_periods_days,
    ]

    if not any(per_location_actions) and not args.accuracy_report and not args.dedupe:
        parser.print_help()
        return

    if args.dedupe:
        result = deduplicate(args.db)
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
                # A handful of locations hitting a transient error is expected
                # from time to time and shouldn't fail the whole run (the other
                # locations' data, the report, and the daily commit are still
                # good and shouldn't be thrown away) - but every location
                # failing points at something systemic (bad credentials, an
                # outage) that's worth actually failing loudly for.
                if len(failed_locations) == len(AUSTRALIAN_LOCATIONS):
                    exit_code = 1
        else:
            if not _run_for_location(args, _location_from_args(args)):
                exit_code = 1

    if args.accuracy_report:
        locations = AUSTRALIAN_LOCATIONS if args.all_locations else [_location_from_args(args)]
        # build_html_report only ever uses the most recent report_history_days
        # anyway (everything older gets trimmed immediately) - +report_window
        # is a small buffer so the centered rolling-MAE trend has full context
        # right at the edge of that window too, matching what an unbounded
        # read would have produced for every visible date.
        window_days = args.report_history_days + args.report_window
        long_df = build_predictions_long(args.db, locations, window_days=window_days)
        # Small-slice sub-daily rain: concatenated in as extra synthetic
        # targets (see build_period_predictions_long's docstring) rather than
        # given a separate report/rendering path.
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

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
