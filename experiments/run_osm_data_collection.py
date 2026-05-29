"""Collect and cache processed OpenStreetMap points for real experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.osm_data import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PLACE,
    collect_osm_points,
    load_processed_points,
    save_processed_data,
)


OUTPUT_DIR = PROJECT_ROOT / DEFAULT_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Download and preprocess OSM points for real experiments."
    )
    parser.add_argument("--place", default=DEFAULT_PLACE, help="OSM place name.")
    parser.add_argument(
        "--max-demand",
        type=int,
        default=None,
        help="Maximum number of demand points to keep.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum number of candidate points to keep.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load existing processed CSV files instead of downloading OSM data.",
    )
    return parser.parse_args()


def print_summary(
    place: str,
    demand_count: int,
    candidate_count: int,
    demand_path: Path,
    candidate_path: Path,
) -> None:
    """Print a concise collection summary."""

    print("OSM data collection summary")
    print(f"Place: {place}")
    print(f"Demand points: {demand_count}")
    print(f"Candidate points: {candidate_count}")
    print(f"Demand CSV: {demand_path}")
    print(f"Candidate CSV: {candidate_path}")


def main() -> int:
    """Run OSM data collection or load cached processed CSV files."""

    args = parse_args()
    demand_path = OUTPUT_DIR / "demand_points.csv"
    candidate_path = OUTPUT_DIR / "candidate_points.csv"

    if args.use_cache:
        try:
            demand_points, candidate_points = load_processed_points(OUTPUT_DIR)
        except FileNotFoundError as exc:
            print(f"Cache not available: {exc}", file=sys.stderr)
            return 1

        print_summary(
            args.place,
            len(demand_points),
            len(candidate_points),
            demand_path,
            candidate_path,
        )
        return 0

    try:
        demand_points, candidate_points = collect_osm_points(
            place_name=args.place,
            max_demand=args.max_demand,
            max_candidates=args.max_candidates,
            seed=args.seed,
        )
    except Exception as exc:
        print(
            "Could not download or preprocess OSM data. "
            "Check the place name, internet connection, and Overpass availability. "
            f"Error: {exc}",
            file=sys.stderr,
        )
        if demand_path.exists() and candidate_path.exists():
            print(
                "Processed CSV cache exists. Re-run with --use-cache to reuse it.",
                file=sys.stderr,
            )
        return 1

    saved_demand_path, saved_candidate_path = save_processed_data(
        demand_points,
        candidate_points,
        OUTPUT_DIR,
    )

    print_summary(
        args.place,
        len(demand_points),
        len(candidate_points),
        saved_demand_path,
        saved_candidate_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
