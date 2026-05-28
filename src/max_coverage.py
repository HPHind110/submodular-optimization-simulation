"""Utilities for the small Maximum Coverage instance from ``SPEC.md``.

This module provides the toy coverage instance and helper functions for
evaluating selected subsets.
"""

from __future__ import annotations

from collections.abc import Mapping


CoverageSets = Mapping[str, set[int]]


def coverage_objective(sets: CoverageSets, selected: set[str]) -> float:
    """Return the number of covered universe elements.

    Args:
        sets: Mapping from item name to the set of covered elements.
        selected: Chosen item names.

    Returns:
        The coverage size of the union of all selected sets.
    """

    covered: set[int] = set()
    for item in selected:
        covered.update(sets[item])
    return float(len(covered))


def coverage_marginal_gain(sets: CoverageSets, x: str, selected: set[str]) -> float:
    """Return the marginal gain of adding ``x`` to ``selected``.

    Args:
        sets: Mapping from item name to the set of covered elements.
        x: Candidate item to add.
        selected: Current chosen item names.

    Returns:
        The increase in coverage after adding ``x``.
    """

    if x in selected:
        return 0.0

    covered_before: set[int] = set()
    for item in selected:
        covered_before.update(sets[item])

    covered_after = covered_before | sets[x]
    return float(len(covered_after) - len(covered_before))


def get_small_coverage_instance() -> tuple[CoverageSets, list[str], int]:
    """Return the Maximum Coverage toy instance defined in ``SPEC.md``.

    Returns:
        A tuple ``(sets, items, k)`` where:
        - ``sets`` maps item names to covered elements
        - ``items`` is the deterministic item order
        - ``k`` is the cardinality budget
    """

    sets: dict[str, set[int]] = {
        "A1": {1, 2, 3},
        "A2": {2, 3, 4},
        "A3": {4, 5},
        "A4": {5, 6, 7},
        "A5": {1, 7},
    }
    items = ["A1", "A2", "A3", "A4", "A5"]
    k = 2
    return sets, items, k
