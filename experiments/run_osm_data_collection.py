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
        default=1000,
        help="Maximum number of demand points to keep.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=400,
        help="Maximum number of candidate points to keep.",
    )
    parser.add_argument(
        "--include-road-nodes",
        action="store_true",
        help="Add sampled OSM road network nodes to candidate locations.",
    )
    parser.add_argument(
        "--max-road-node-candidates",
        type=int,
        default=400,
        help="Maximum number of sampled road network nodes to add as candidates.",
    )
    parser.add_argument(
        "--network-type",
        default="walk",
        help="OSMnx network_type for road node candidates.",
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
    candidate_points=None,
) -> None:
    """Print a concise collection summary."""

    print("OSM data collection summary")
    print(f"Place: {place}")
    print(f"Demand points: {demand_count}")
    print(f"Candidate points: {candidate_count}")
    if candidate_points is not None and "source_type" in candidate_points:
        print("Candidate source_type counts:")
        counts = candidate_points["source_type"].value_counts().sort_index()
        for source_type, count in counts.items():
            print(f"  {source_type}: {count}")
    print(f"Demand CSV: {demand_path}")
    print(f"Candidate CSV: {candidate_path}")


def save_candidate_scenario_files(args: argparse.Namespace) -> tuple[Path, Path]:
    """Create separate bus-stop-only and road-node candidate CSV files."""

    _, bus_stop_candidates = collect_osm_points(
        place_name=args.place,
        max_demand=args.max_demand,
        max_candidates=args.max_candidates,
        seed=args.seed,
        include_road_nodes=False,
    )
    _, road_node_candidates = collect_osm_points(
        place_name=args.place,
        max_demand=args.max_demand,
        max_candidates=args.max_candidates,
        seed=args.seed,
        include_road_nodes=True,
        max_road_node_candidates=args.max_road_node_candidates,
        network_type=args.network_type,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bus_stop_path = OUTPUT_DIR / "candidate_points_bus_stop_only.csv"
    road_node_path = OUTPUT_DIR / "candidate_points_road_nodes.csv"
    bus_stop_candidates.to_csv(bus_stop_path, index=False)
    road_node_candidates.to_csv(road_node_path, index=False)
    return bus_stop_path, road_node_path


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
            candidate_points,
        )
        return 0

    try:
        demand_points, candidate_points = collect_osm_points(
            place_name=args.place,
            max_demand=args.max_demand,
            max_candidates=args.max_candidates,
            seed=args.seed,
            include_road_nodes=args.include_road_nodes,
            max_road_node_candidates=args.max_road_node_candidates,
            network_type=args.network_type,
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
    bus_stop_path, road_node_path = save_candidate_scenario_files(args)

    print_summary(
        args.place,
        len(demand_points),
        len(candidate_points),
        saved_demand_path,
        saved_candidate_path,
        candidate_points,
    )
    print(f"Bus-stop-only candidate CSV: {bus_stop_path}")
    print(f"Road-node candidate CSV: {road_node_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
