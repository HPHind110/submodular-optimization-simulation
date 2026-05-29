"""Validate candidate scenario comparison outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

REQUIRED_COLUMNS = [
    "scenario",
    "n_demand",
    "n_candidate",
    "radius_m",
    "k",
    "algorithm",
    "coverage_count",
    "coverage_rate",
    "avg_nearest_distance_m",
    "max_nearest_distance_m",
    "eval_count",
    "runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Validate candidate scenario outputs.")
    parser.add_argument(
        "--radii",
        nargs="+",
        type=int,
        default=[150],
        help="Radius labels to validate.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    """Raise a validation error."""

    raise ValueError(message)


def validate_table(table: pd.DataFrame, radius: int) -> None:
    """Validate one candidate scenario table."""

    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        fail(f"R{radius}: missing columns {missing}.")

    scenarios = set(table["scenario"])
    if scenarios != {"bus_stop_only", "road_nodes"}:
        fail(f"R{radius}: unexpected scenarios {sorted(scenarios)}.")

    algorithms = set(table["algorithm"])
    if algorithms != {"Greedy", "Lazy Greedy"}:
        fail(f"R{radius}: unexpected algorithms {sorted(algorithms)}.")

    if not np.all(table["radius_m"].to_numpy(dtype=int) == radius):
        fail(f"R{radius}: radius_m column does not match filename.")

    coverage_count = table["coverage_count"].to_numpy(dtype=np.float64)
    n_demand = table["n_demand"].to_numpy(dtype=np.float64)
    if np.any(coverage_count < 0) or np.any(coverage_count > n_demand):
        fail(f"R{radius}: coverage_count outside [0, n_demand].")

    expected_rate = coverage_count / n_demand
    if not np.allclose(table["coverage_rate"].to_numpy(dtype=np.float64), expected_rate):
        fail(f"R{radius}: coverage_rate formula validation failed.")

    if np.any(table["eval_count"].to_numpy(dtype=int) <= 0):
        fail(f"R{radius}: eval_count must be positive.")
    if np.any(table["runtime_seconds"].to_numpy(dtype=np.float64) < 0):
        fail(f"R{radius}: runtime_seconds must be non-negative.")

    for (scenario, k), subset in table.groupby(["scenario", "k"], sort=False):
        greedy = subset[subset["algorithm"] == "Greedy"]
        lazy = subset[subset["algorithm"] == "Lazy Greedy"]
        if greedy.empty or lazy.empty:
            fail(f"R{radius}: missing Greedy/Lazy row for {scenario}, k={k}.")
        if int(lazy.iloc[0]["coverage_count"]) != int(greedy.iloc[0]["coverage_count"]):
            fail(f"R{radius}: Lazy coverage differs from Greedy for {scenario}, k={k}.")
        if int(lazy.iloc[0]["eval_count"]) > int(greedy.iloc[0]["eval_count"]):
            fail(f"R{radius}: Lazy eval_count exceeds Greedy for {scenario}, k={k}.")


def main() -> int:
    """Validate candidate scenario CSV files."""

    args = parse_args()
    try:
        for radius in args.radii:
            path = OUTPUT_TABLES_DIR / f"candidate_scenario_comparison_R{radius}.csv"
            if not path.exists():
                fail(f"Missing candidate scenario CSV: {path}")
            validate_table(pd.read_csv(path), radius)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("ALL CANDIDATE SCENARIO VALIDATIONS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
