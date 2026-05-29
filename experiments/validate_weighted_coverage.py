"""Validate weighted coverage scenario outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "outputs" / "tables" / "weighted_coverage_scenarios.csv"
DEMAND_PATH = PROJECT_ROOT / "data" / "processed" / "demand_points.csv"

REQUIRED_COLUMNS = [
    "weight_scheme",
    "k",
    "algorithm",
    "coverage_count",
    "coverage_rate",
    "total_weight",
    "weighted_coverage_value",
    "weighted_coverage_rate",
    "priority_points_total",
    "priority_points_covered",
    "priority_coverage_rate",
    "eval_count",
    "runtime_seconds",
    "selected_ids",
]


def fail(message: str) -> None:
    """Raise a validation error with a concise message."""

    raise ValueError(message)


def validate_required_columns(table: pd.DataFrame) -> None:
    """Validate that all required columns exist."""

    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        fail(f"Missing required columns: {missing}")


def validate_row_ranges(table: pd.DataFrame) -> None:
    """Validate per-row numeric ranges and rate formulas."""

    if "n_demand" in table.columns:
        n_demand = table["n_demand"].to_numpy(dtype=np.float64)
    elif DEMAND_PATH.exists():
        n_demand = np.full(len(table), float(len(pd.read_csv(DEMAND_PATH))))
    else:
        n_demand = np.full(len(table), float(table["coverage_count"].max()))

    coverage_count = table["coverage_count"].to_numpy(dtype=np.float64)
    if np.any(coverage_count < 0) or np.any(coverage_count > n_demand):
        fail("coverage_count must lie in [0, n_demand].")

    total_weight = table["total_weight"].to_numpy(dtype=np.float64)
    weighted_value = table["weighted_coverage_value"].to_numpy(dtype=np.float64)
    if np.any(total_weight <= 0):
        fail("total_weight must be positive.")
    if np.any(weighted_value < 0) or np.any(weighted_value - total_weight > 1e-9):
        fail("weighted_coverage_value must lie in [0, total_weight].")

    expected_weighted_rate = weighted_value / total_weight
    if not np.allclose(
        table["weighted_coverage_rate"].to_numpy(dtype=np.float64),
        expected_weighted_rate,
    ):
        fail("weighted_coverage_rate formula validation failed.")

    priority_total = table["priority_points_total"].to_numpy(dtype=np.float64)
    priority_covered = table["priority_points_covered"].to_numpy(dtype=np.float64)
    if np.any(priority_total < 0):
        fail("priority_points_total must be non-negative.")
    if np.any(priority_covered < 0) or np.any(priority_covered - priority_total > 1e-9):
        fail("priority_points_covered must lie in [0, priority_points_total].")

    expected_priority_rate = np.divide(
        priority_covered,
        priority_total,
        out=np.zeros_like(priority_covered),
        where=priority_total > 0,
    )
    if not np.allclose(
        table["priority_coverage_rate"].to_numpy(dtype=np.float64),
        expected_priority_rate,
    ):
        fail("priority_coverage_rate formula validation failed.")


def validate_monotonic_by_k(table: pd.DataFrame) -> None:
    """Validate weighted coverage value does not decrease as k increases."""

    grouped = table.groupby(["weight_scheme", "algorithm"], sort=False)
    for (weight_scheme, algorithm), subset in grouped:
        ordered = subset.sort_values("k")
        values = ordered["weighted_coverage_value"].to_numpy(dtype=np.float64)
        if np.any(np.diff(values) < -1e-9):
            fail(
                "weighted_coverage_value decreases with k for "
                f"weight_scheme={weight_scheme}, algorithm={algorithm}."
            )


def validate_lazy_against_greedy(table: pd.DataFrame) -> None:
    """Validate Lazy Greedy value/evaluation relationships against Greedy."""

    grouped = table.groupby(["k", "weight_scheme"], sort=False)
    for (k, weight_scheme), subset in grouped:
        greedy_rows = subset[subset["algorithm"] == "Greedy"]
        lazy_rows = subset[subset["algorithm"] == "Lazy Greedy"]
        if greedy_rows.empty or lazy_rows.empty:
            fail(f"Missing Greedy or Lazy Greedy row for k={k}, scheme={weight_scheme}.")

        greedy_row = greedy_rows.iloc[0]
        lazy_row = lazy_rows.iloc[0]
        if not np.isclose(
            float(lazy_row["weighted_coverage_value"]),
            float(greedy_row["weighted_coverage_value"]),
        ):
            fail(
                "Lazy Greedy weighted_coverage_value differs from Greedy for "
                f"k={k}, scheme={weight_scheme}."
            )
        if int(lazy_row["eval_count"]) > int(greedy_row["eval_count"]):
            fail(
                "Lazy Greedy eval_count exceeds Greedy for "
                f"k={k}, scheme={weight_scheme}."
            )


def main() -> int:
    """Validate weighted coverage scenario CSV."""

    if not CSV_PATH.exists():
        print(f"Missing weighted coverage scenario CSV: {CSV_PATH}", file=sys.stderr)
        return 1

    try:
        table = pd.read_csv(CSV_PATH)
        validate_required_columns(table)
        validate_row_ranges(table)
        validate_monotonic_by_k(table)
        validate_lazy_against_greedy(table)
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("ALL WEIGHTED COVERAGE VALIDATIONS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
