from __future__ import annotations

import argparse
import json
from pathlib import Path

from weather_ensemble.config import (
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Weather ensemble forecaster")
    parser.add_argument("--lat", type=float, help="Latitude")
    parser.add_argument("--lon", type=float, help="Longitude")
    parser.add_argument("--name", help="Location name")
    parser.add_argument("--timezone", help="Timezone, e.g. Australia/Melbourne")
    parser.add_argument("--db", type=Path, default=get_db_path(), help="SQLite DB path")
    parser.add_argument("--window", type=int, default=get_rolling_window_days())

    parser.add_argument("--collect", action="store_true", help="Collect forecasts for tomorrow")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    location = _location_from_args(args)

    if args.deploy_phases:
        result = deploy_all_phases(
            db_path=args.db,
            location=location,
            days_back=args.deploy_phases,
            processed_path=Path("data/processed/features.parquet"),
            model_dir=args.model_dir,
        )
        _print_json(result)
        return

    if args.backfill:
        backfill(args.db, location, args.backfill)
        print(f"Backfilled {args.backfill} days for {location.name} into {args.db}")

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
        result = train_models(args.db, location, args.model_dir)
        _print_json(result)

    if args.predict_ml:
        result = predict_latest_ml(args.db, location, args.model_dir)
        _print_json(result)

    if not any(
        [
            args.backfill,
            args.collect,
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


if __name__ == "__main__":
    main()
