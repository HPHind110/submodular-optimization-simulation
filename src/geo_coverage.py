"""Maximum Coverage helpers for geospatial experiments."""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
CoverageSets = dict[int, set[int]]


def build_coverage_sets(distance_matrix: FloatArray, radius: float) -> CoverageSets:
    """Build demand coverage sets for each candidate facility.

    Args:
        distance_matrix: Array with shape ``(n_demand, n_candidate)``.
        radius: Coverage radius in meters.

    Returns:
        A mapping from candidate index to the set of demand indices within
        ``radius`` meters.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    distances = np.asarray(distance_matrix, dtype=np.float64)
    if distances.ndim != 2:
        raise ValueError("distance_matrix must be a 2D array.")

    coverage_sets: CoverageSets = {}
    for candidate_index in range(distances.shape[1]):
        covered = np.flatnonzero(distances[:, candidate_index] <= radius)
        coverage_sets[candidate_index] = set(int(index) for index in covered)
    return coverage_sets


def covered_demand_indices(
    coverage_sets: Mapping[int, set[int]],
    selected: Collection[int],
) -> set[int]:
    """Return all demand indices covered by the selected candidates."""

    covered: set[int] = set()
    for candidate_index in selected:
        covered.update(coverage_sets.get(candidate_index, set()))
    return covered


def coverage_objective_geo(
    coverage_sets: Mapping[int, set[int]],
    selected: Collection[int],
    n_demand: int,
) -> int:
    """Return the number of demand points covered by ``selected`` candidates."""

    if n_demand < 0:
        raise ValueError("n_demand must be non-negative.")

    covered = covered_demand_indices(coverage_sets, selected)
    return min(len(covered), n_demand)


def coverage_marginal_gain_geo(
    coverage_sets: Mapping[int, set[int]],
    x: int,
    selected: Collection[int],
) -> int:
    """Return the number of newly covered demand points after adding ``x``."""

    if x in selected:
        return 0

    covered_before = covered_demand_indices(coverage_sets, selected)
    covered_after = covered_before | coverage_sets.get(x, set())
    return len(covered_after) - len(covered_before)
