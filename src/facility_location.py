"""Utilities for small Facility Location experiments.

This module provides synthetic clustered data generation, Gaussian similarity,
and objective helpers for the facility location set function.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def gaussian_similarity(X: FloatArray, sigma: float = 1.0) -> FloatArray:
    """Compute the Gaussian similarity matrix for a set of 2D points.

    Args:
        X: Array of shape ``(n_samples, n_features)``.
        sigma: Positive bandwidth parameter.

    Returns:
        A square similarity matrix ``W`` where
        ``W[i, j] = exp(-||x_i - x_j||^2 / (2 * sigma^2))``.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    points = np.asarray(X, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("X must be a 2D array.")

    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    squared_distances = np.sum(diffs * diffs, axis=2)
    return np.exp(-squared_distances / (2.0 * sigma * sigma))


def facility_objective(W: FloatArray, selected: set[int]) -> float:
    """Evaluate the facility location objective ``f(S)``.

    Args:
        W: Similarity matrix of shape ``(n_samples, n_samples)``.
        selected: Chosen representative indices.

    Returns:
        The value ``sum_i max_{j in selected} W[i, j]``. If ``selected`` is
        empty, the objective value is ``0.0``.
    """

    if not selected:
        return 0.0

    indices = sorted(selected)
    max_similarities = np.max(W[:, indices], axis=1)
    return float(np.sum(max_similarities))


def facility_marginal_gain(W: FloatArray, x: int, selected: set[int]) -> float:
    """Return the marginal gain of adding index ``x`` to ``selected``.

    Args:
        W: Similarity matrix of shape ``(n_samples, n_samples)``.
        x: Candidate representative index.
        selected: Current representative set.

    Returns:
        The improvement in the facility location objective after adding ``x``.
    """

    if x in selected:
        return 0.0

    n_samples = W.shape[0]
    if x < 0 or x >= n_samples:
        raise IndexError("x is out of bounds for the similarity matrix.")

    if not selected:
        current_best = np.zeros(n_samples, dtype=np.float64)
    else:
        current_best = np.max(W[:, sorted(selected)], axis=1)

    updated_best = np.maximum(current_best, W[:, x])
    return float(np.sum(updated_best) - np.sum(current_best))


def generate_cluster_data(
    n_per_cluster: int = 5,
    centers: Sequence[Sequence[float]] | None = None,
    scale: float = 0.4,
    seed: int = 42,
) -> tuple[FloatArray, IntArray]:
    """Generate a small 2D Gaussian cluster dataset.

    Args:
        n_per_cluster: Number of points to sample for each cluster.
        centers: Cluster centers. When omitted, two centers are used.
        scale: Standard deviation of each Gaussian cluster.
        seed: Random seed for reproducibility.

    Returns:
        A tuple ``(X, labels)`` where ``X`` has shape ``(n_points, 2)`` and
        ``labels`` stores the cluster id for each point.
    """

    if n_per_cluster <= 0:
        raise ValueError("n_per_cluster must be positive.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    default_centers = np.array([[-2.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    center_array = (
        default_centers
        if centers is None
        else np.asarray(centers, dtype=np.float64)
    )

    if center_array.ndim != 2 or center_array.shape[1] != 2:
        raise ValueError("centers must describe 2D points.")

    rng = np.random.default_rng(seed)
    point_blocks: list[FloatArray] = []
    label_blocks: list[IntArray] = []

    for label, center in enumerate(center_array):
        samples = rng.normal(loc=center, scale=scale, size=(n_per_cluster, 2))
        point_blocks.append(samples.astype(np.float64))
        label_blocks.append(np.full(n_per_cluster, label, dtype=np.int64))

    X = np.vstack(point_blocks)
    labels = np.concatenate(label_blocks)
    return X, labels
