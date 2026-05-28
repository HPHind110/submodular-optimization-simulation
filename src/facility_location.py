"""Problem definition skeleton for Facility Location experiments.

This module will later contain:

- synthetic clustered data generation
- similarity matrix construction
- facility location objective helpers
"""

from __future__ import annotations

from typing import Any


def generate_small_dataset(
    n_points: int = 30,
    n_clusters: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Return a placeholder description for a small clustered dataset."""

    return {
        "n_points": n_points,
        "n_clusters": n_clusters,
        "seed": seed,
        "status": "not_implemented",
    }


def facility_location_objective(selected_indices: list[int], similarity_matrix: Any) -> float:
    """Return a placeholder objective value for facility location."""

    _ = selected_indices
    _ = similarity_matrix
    return 0.0
