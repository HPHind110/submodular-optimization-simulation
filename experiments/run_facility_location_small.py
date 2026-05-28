"""Run a small Facility Location experiment from the SPEC.

This script is intended to:

- generate a small clustered dataset
- compare brute force, greedy, and random baseline
- export summary tables and a scatter plot
"""

from __future__ import annotations

from src.facility_location import generate_small_dataset


def main() -> None:
    """Entry point for the small Facility Location experiment."""

    dataset = generate_small_dataset()
    print("Facility Location small experiment skeleton")
    print(
        f"Configured points: {dataset['n_points']}, "
        f"clusters: {dataset['n_clusters']}"
    )


if __name__ == "__main__":
    main()
