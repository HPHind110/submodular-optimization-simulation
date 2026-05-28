"""Core algorithm skeletons for submodular optimization experiments.

This module defines lightweight placeholders for the algorithms required by
the project specification:

- brute force
- greedy
- lazy greedy
- stochastic greedy

The current version only provides clear function signatures, docstrings, and
minimal return structures so the project layout is ready for later
implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


ObjectiveFunction = Callable[[Sequence[Any]], float]
MarginalGainFunction = Callable[[Sequence[Any], Any], float]


def brute_force(
    items: Sequence[Any],
    k: int,
    objective_fn: ObjectiveFunction,
) -> dict[str, Any]:
    """Return a placeholder result for brute-force search.

    Args:
        items: Candidate items to choose from.
        k: Number of items to select.
        objective_fn: Objective function evaluated on a chosen subset.

    Returns:
        A result dictionary with the fields expected by later experiments.
    """

    return {
        "algorithm": "brute_force",
        "selected_set": [],
        "objective_value": 0.0,
        "evaluations": 0,
        "runtime_seconds": 0.0,
        "status": "not_implemented",
    }


def greedy(
    items: Sequence[Any],
    k: int,
    marginal_gain_fn: MarginalGainFunction,
) -> dict[str, Any]:
    """Return a placeholder result for the standard greedy algorithm."""

    return {
        "algorithm": "greedy",
        "selected_set": [],
        "objective_value": 0.0,
        "evaluations": 0,
        "runtime_seconds": 0.0,
        "status": "not_implemented",
    }


def lazy_greedy(
    items: Sequence[Any],
    k: int,
    marginal_gain_fn: MarginalGainFunction,
) -> dict[str, Any]:
    """Return a placeholder result for lazy greedy with a priority queue."""

    return {
        "algorithm": "lazy_greedy",
        "selected_set": [],
        "objective_value": 0.0,
        "evaluations": 0,
        "runtime_seconds": 0.0,
        "status": "not_implemented",
    }


def stochastic_greedy(
    items: Sequence[Any],
    k: int,
    marginal_gain_fn: MarginalGainFunction,
    epsilon: float,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return a placeholder result for stochastic greedy.

    Args:
        items: Candidate items to choose from.
        k: Number of items to select.
        marginal_gain_fn: Marginal gain function for candidate evaluation.
        epsilon: Sampling parameter from the specification.
        seed: Optional random seed for reproducibility.
    """

    return {
        "algorithm": "stochastic_greedy",
        "selected_set": [],
        "objective_value": 0.0,
        "evaluations": 0,
        "runtime_seconds": 0.0,
        "epsilon": epsilon,
        "seed": seed,
        "status": "not_implemented",
    }
