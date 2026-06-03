from __future__ import annotations

import argparse
from pathlib import Path

from weather_ensemble.config import Location, get_db_path, get_default_location, get_rolling_window_days
from weather_ensemble.service import backfill, blend_forecast, collect_forecasts, export_modelling_table, record_actual


def _location_from_args(args: argparse.Namespace) -> Location:
    default = get_default_location()
    return Location(
        name=args.name or default.name,
        lat=args.lat if args.lat is not None else default.lat,
        lon=args.lon if args.lon is not None else default.lon,
        timezone=args.timezone or default.timezone,
    )


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
    parser.add_argument("--forecast", action="store_true", help="Generate blended forecast")
    parser.add_argument("--all", action="store_true", help="Run collect, record actual, and forecast")
    parser.add_argument("--export", type=Path, help="Export modelling table to parquet")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    location = _location_from_args(args)

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
        print(result)

    if args.export:
        output = export_modelling_table(args.db, location, args.export)
        print(f"Exported modelling table to {output}")

    if not any([args.backfill, args.collect, args.record_actual, args.forecast, args.all, args.export]):
        parser.print_help()


if __name__ == "__main__":
    main()
