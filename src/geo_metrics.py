"""Geospatial distance metrics for real-world experiments."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def _as_xy_array(points: FloatArray, name: str) -> FloatArray:
    """Validate and convert an input array to shape ``(n_points, 2)``."""

    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must be a 2D array with shape (n_points, 2).")
    return array


def _selected_indices(selected: Collection[int], n_candidates: int) -> list[int]:
    """Validate selected candidate indices and return a stable list."""

    indices = sorted(set(selected))
    if not indices:
        raise ValueError("selected must contain at least one candidate index.")
    if indices[0] < 0 or indices[-1] >= n_candidates:
        raise IndexError("selected contains a candidate index outside distance_matrix.")
    return indices


def pairwise_distance_matrix(
    demand_xy: FloatArray,
    candidate_xy: FloatArray,
) -> FloatArray:
    """Return Euclidean distances in meters from demand to candidate points.

    Args:
        demand_xy: Projected demand coordinates with shape ``(n_demand, 2)``.
        candidate_xy: Projected candidate coordinates with shape
            ``(n_candidate, 2)``.

    Returns:
        A distance matrix with shape ``(n_demand, n_candidate)`` where entry
        ``[i, j]`` is the Euclidean distance from demand point ``i`` to
        candidate point ``j``.
    """

    demand = _as_xy_array(demand_xy, "demand_xy")
    candidates = _as_xy_array(candidate_xy, "candidate_xy")
    differences = demand[:, np.newaxis, :] - candidates[np.newaxis, :, :]
    return np.linalg.norm(differences, axis=2)


def nearest_distance_to_selected(
    distance_matrix: FloatArray,
    selected: Collection[int],
) -> FloatArray:
    """Return each demand point's distance to the nearest selected candidate.

    Args:
        distance_matrix: Array with shape ``(n_demand, n_candidate)``.
        selected: Candidate indices selected as facilities.

    Returns:
        A vector with shape ``(n_demand,)`` containing nearest selected
        candidate distances.

    Raises:
        ValueError: If ``selected`` is empty or ``distance_matrix`` is not 2D.
        IndexError: If a selected index is outside the candidate dimension.
    """

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array.")

    indices = _selected_indices(selected, distances.shape[1])
    return np.min(distances[:, indices], axis=1)


def average_nearest_distance(
    distance_matrix: FloatArray,
    selected: Collection[int],
) -> float:
    """Return the average demand distance to the nearest selected candidate."""

    nearest_distances = nearest_distance_to_selected(distance_matrix, selected)
    return float(np.mean(nearest_distances))


def max_nearest_distance(
    distance_matrix: FloatArray,
    selected: Collection[int],
) -> float:
    """Return the maximum demand distance to the nearest selected candidate."""

    nearest_distances = nearest_distance_to_selected(distance_matrix, selected)
    return float(np.max(nearest_distances))
