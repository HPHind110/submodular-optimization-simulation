"""Facility Location helpers for geospatial experiments."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def build_similarity_matrix(distance_matrix: FloatArray, sigma: float) -> FloatArray:
    """Build a Gaussian similarity matrix from distances in meters.

    Args:
        distance_matrix: Array with shape ``(n_demand, n_candidate)``.
        sigma: Positive Gaussian bandwidth in meters.

    Returns:
        Matrix ``W`` where ``W[i, j] = exp(-d(i, j)^2 / (2 sigma^2))``.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array.")

    return np.exp(-(distances * distances) / (2.0 * sigma * sigma))


def _selected_indices(selected: Collection[int], n_candidates: int) -> list[int]:
    """Validate selected candidate indices and return a stable list."""

    indices = sorted(set(selected))
    if not indices:
        return []
    if indices[0] < 0 or indices[-1] >= n_candidates:
        raise IndexError("selected contains a candidate index outside the matrix.")
    return indices


def facility_objective_geo(W: FloatArray, selected: Collection[int]) -> float:
    """Evaluate the geospatial Facility Location objective."""

    similarity = np.asarray(W, dtype=np.float64)
    if similarity.ndim != 2:
        raise ValueError("W must be a 2D array.")

    indices = _selected_indices(selected, similarity.shape[1])
    if not indices:
        return 0.0

    return float(np.sum(np.max(similarity[:, indices], axis=1)))


def facility_marginal_gain_geo(
    W: FloatArray,
    x: int,
    selected: Collection[int],
) -> float:
    """Return the objective gain after adding candidate ``x`` to ``selected``."""

    similarity = np.asarray(W, dtype=np.float64)
    if similarity.ndim != 2:
        raise ValueError("W must be a 2D array.")
    if x < 0 or x >= similarity.shape[1]:
        raise IndexError("x is outside the candidate dimension of W.")
    if x in selected:
        return 0.0

    indices = _selected_indices(selected, similarity.shape[1])
    if not indices:
        current_best = np.zeros(similarity.shape[0], dtype=np.float64)
    else:
        current_best = np.max(similarity[:, indices], axis=1)

    updated_best = np.maximum(current_best, similarity[:, x])
    return float(np.sum(updated_best) - np.sum(current_best))


def selected_assignment(
    distance_matrix: FloatArray,
    selected: Collection[int],
) -> IntArray:
    """Return the nearest selected candidate index for each demand point."""

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array.")

    indices = _selected_indices(selected, distances.shape[1])
    if not indices:
        raise ValueError("selected must contain at least one candidate index.")

    nearest_positions = np.argmin(distances[:, indices], axis=1)
    return np.asarray([indices[position] for position in nearest_positions], dtype=np.int64)
