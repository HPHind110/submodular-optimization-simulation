"""Problem definition skeleton for the Maximum Coverage experiment.

This module will later contain:

- the toy universe and candidate subsets from the specification
- objective and marginal gain utilities
- optional random baseline helpers
"""

from __future__ import annotations

from typing import Any


def build_small_instance() -> dict[str, Any]:
    """Return the small Maximum Coverage example defined in the SPEC."""

    universe = {1, 2, 3, 4, 5, 6, 7}
    subsets = {
        "A1": {1, 2, 3},
        "A2": {2, 3, 4},
        "A3": {4, 5},
        "A4": {5, 6, 7},
        "A5": {1, 7},
    }
    k = 2
    return {"universe": universe, "subsets": subsets, "k": k}


def coverage_objective(selected_sets: list[set[int]]) -> int:
    """Compute covered elements for a chosen collection of subsets.

    This placeholder implementation is intentionally simple and can be reused
    by later experiment scripts.
    """

    covered: set[int] = set()
    for subset in selected_sets:
        covered.update(subset)
    return len(covered)
