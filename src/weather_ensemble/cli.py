from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from weather_ensemble.config import (
    AUSTRALIAN_LOCATIONS,
    Location,
    get_db_path,
    get_default_location,
    get_rolling_window_days,
)
from weather_ensemble.ml import build_feature_table, predict_latest_ml, train_models
from weather_ensemble.phases import deploy_all_phases
from weather_ensemble.service import (
    backfill,
    blend_forecast,
    collect_forecasts,
    collect_open_meteo_only,
    export_modelling_table,
    record_actual,
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
    parser.add_argument("--backfill", type=int, metavar="DAYS", help="Backfill forecasts and actuals")
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
        "--all-locations",
        action="store_true",
        help="Repeat the requested actions for every location in AUSTRALIAN_LOCATIONS "
        "instead of a single --lat/--lon location",
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
    except Exception:
        fallback = path.with_suffix(".csv")
        df.to_csv(fallback, index=False)
        return fallback


def _run_for_location(args: argparse.Namespace, location: Location) -> None:
    model_dir = _location_model_dir(args.model_dir, location)

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
        return

    if args.backfill:
        backfill(args.db, location, args.backfill)
        print(f"Backfilled {args.backfill} days for {location.name} into {args.db}")

    if args.collect_open_meteo:
        records = collect_open_meteo_only(args.db, location)
        print(f"Collected {len(records)} Open-Meteo model forecast records")

    if args.collect or args.all:
        records = collect_forecasts(args.db, location)
        print(f"Collected {len(records)} forecast records")

    if args.record_actual or args.all:
        record_actual(args.db, location)
        print("Recorded yesterday's actual weather")

    if args.forecast or args.all:
        result = blend_forecast(args.db, location, args.window)
        _print_json(result)

    if args.export:
        output = export_modelling_table(args.db, location, args.export)
        print(f"Exported long modelling table to {output}")

    if args.build_dataset:
        df = build_feature_table(args.db, location)
        output = _write_dataframe(df, args.build_dataset)
        print(f"Exported wide ML feature table to {output} ({len(df)} rows, {len(df.columns)} columns)")

    if args.train:
        result = train_models(args.db, location, model_dir, window_days=args.train_window)
        _print_json(result)

    if args.predict_ml:
        result = predict_latest_ml(args.db, location, model_dir)
        _print_json(result)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any(
        [
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
        ]
    ):
        parser.print_help()
        return

    if args.all_locations:
        for location in AUSTRALIAN_LOCATIONS:
            print(f"=== {location.name} ===")
            _run_for_location(args, location)
    else:
        _run_for_location(args, _location_from_args(args))


if __name__ == "__main__":
    main()
