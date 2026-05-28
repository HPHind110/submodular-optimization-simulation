"""Core algorithms for small submodular optimization experiments.

This module implements the algorithms required by ``SPEC.md``:

- brute force
- greedy
- random baseline
- lazy greedy
- stochastic greedy

The functions are written to be lightweight and reusable by standalone
experiment scripts. They do not print intermediate output.
"""

from __future__ import annotations

import heapq
import math
import random
import time
from collections.abc import Callable, Hashable, Sequence
from itertools import combinations
from typing import TypeVar


T = TypeVar("T", bound=Hashable)
Objective = Callable[[set[T]], float]
MarginalGain = Callable[[set[T], T], float]
Result = dict[str, float | int | set[T]]


def _prepare_items(items: Sequence[T], k: int) -> tuple[T, ...]:
    """Validate the candidate items and return them as an immutable tuple."""

    if k < 0:
        raise ValueError("k must be non-negative.")

    items_tuple = tuple(items)
    if k > len(items_tuple):
        raise ValueError("k cannot be larger than the number of items.")

    if len(set(items_tuple)) != len(items_tuple):
        raise ValueError("items must be unique because selected is returned as a set.")

    return items_tuple


def _build_result(
    selected: set[T],
    value: float,
    eval_count: int,
    start_time: float,
) -> Result:
    """Create a standardized result dictionary for algorithm outputs."""

    runtime = time.perf_counter() - start_time
    return {
        "selected": set(selected),
        "value": value,
        "eval_count": eval_count,
        "runtime": runtime,
    }


def brute_force(items: Sequence[T], k: int, objective: Objective[T]) -> Result:
    """Search all size-``k`` subsets and return the best one.

    Args:
        items: Candidate items to choose from.
        k: Number of items to select.
        objective: Set function to maximize.

    Returns:
        A dictionary with keys ``selected``, ``value``, ``eval_count``,
        and ``runtime``.
    """

    items_tuple = _prepare_items(items, k)
    start_time = time.perf_counter()
    best_selected: set[T] = set()
    best_value = float("-inf")
    eval_count = 0

    for combo in combinations(items_tuple, k):
        candidate = set(combo)
        value = objective(candidate)
        eval_count += 1
        if value > best_value:
            best_selected = candidate
            best_value = value

    return _build_result(best_selected, best_value, eval_count, start_time)


def greedy(
    items: Sequence[T],
    k: int,
    objective: Objective[T],
    marginal_gain: MarginalGain[T] | None = None,
) -> Result:
    """Run the standard greedy algorithm for cardinality-constrained selection.

    Args:
        items: Candidate items to choose from.
        k: Number of items to select.
        objective: Set function to maximize.
        marginal_gain: Optional function that returns the marginal gain of
            adding one item to the current selected set.

    Returns:
        A dictionary with keys ``selected``, ``value``, ``eval_count``,
        and ``runtime``.
    """

    items_tuple = _prepare_items(items, k)
    start_time = time.perf_counter()
    selected: set[T] = set()
    eval_count = 0

    if marginal_gain is None:
        current_value = objective(set())
        eval_count += 1
    else:
        current_value = 0.0

    for _ in range(k):
        best_item: T | None = None
        best_gain = float("-inf")

        for item in items_tuple:
            if item in selected:
                continue

            if marginal_gain is None:
                candidate_value = objective(selected | {item})
                eval_count += 1
                gain = candidate_value - current_value
            else:
                gain = marginal_gain(selected, item)
                eval_count += 1

            if gain > best_gain:
                best_gain = gain
                best_item = item

        if best_item is None:
            break

        selected.add(best_item)

        if marginal_gain is None:
            current_value += best_gain

    if marginal_gain is not None:
        current_value = objective(selected)
        eval_count += 1

    return _build_result(selected, current_value, eval_count, start_time)


def random_baseline(
    items: Sequence[T],
    k: int,
    objective: Objective[T],
    n_trials: int = 1000,
    seed: int = 42,
) -> Result:
    """Evaluate random size-``k`` subsets and keep the best one found.

    Args:
        items: Candidate items to choose from.
        k: Number of items to select.
        objective: Set function to maximize.
        n_trials: Number of random subsets to evaluate.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with keys ``selected``, ``value``, ``eval_count``,
        and ``runtime``.
    """

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    items_tuple = _prepare_items(items, k)
    start_time = time.perf_counter()
    rng = random.Random(seed)

    if k == 0 or k == len(items_tuple):
        selected = set(items_tuple[:k])
        value = objective(selected)
        return _build_result(selected, value, 1, start_time)

    best_selected: set[T] = set()
    best_value = float("-inf")
    eval_count = 0

    for _ in range(n_trials):
        candidate = set(rng.sample(items_tuple, k))
        value = objective(candidate)
        eval_count += 1
        if value > best_value:
            best_selected = candidate
            best_value = value

    return _build_result(best_selected, best_value, eval_count, start_time)


def lazy_greedy(items: Sequence[T], k: int, objective: Objective[T]) -> Result:
    """Run the lazy greedy algorithm using a max-heap of upper bounds.

    Args:
        items: Candidate items to choose from.
        k: Number of items to select.
        objective: Set function to maximize.

    Returns:
        A dictionary with keys ``selected``, ``value``, ``eval_count``,
        and ``runtime``.
    """

    items_tuple = _prepare_items(items, k)
    start_time = time.perf_counter()
    selected: set[T] = set()
    eval_count = 0
    current_value = objective(set())
    eval_count += 1
    heap: list[tuple[float, int, T]] = []

    for index, item in enumerate(items_tuple):
        gain = objective({item}) - current_value
        eval_count += 1
        heapq.heappush(heap, (-gain, index, item))

    while len(selected) < k and heap:
        neg_upper_bound, index, item = heapq.heappop(heap)
        if item in selected:
            continue

        exact_value = objective(selected | {item})
        eval_count += 1
        exact_gain = exact_value - current_value

        if not heap or exact_gain >= -heap[0][0]:
            selected.add(item)
            current_value = exact_value
            continue

        heapq.heappush(heap, (-exact_gain, index, item))

    return _build_result(selected, current_value, eval_count, start_time)


def stochastic_greedy(
    items: Sequence[T],
    k: int,
    objective: Objective[T],
    epsilon: float = 0.1,
    seed: int = 42,
) -> Result:
    """Run stochastic greedy with a random candidate sample at each step.

    The sample size follows the specification:

    ``r = ceil((n / k) * ln(1 / epsilon))``

    Args:
        items: Candidate items to choose from.
        k: Number of items to select.
        objective: Set function to maximize.
        epsilon: Sampling parameter in ``(0, 1)``.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary with keys ``selected``, ``value``, ``eval_count``,
        and ``runtime``.
    """

    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1.")

    items_tuple = _prepare_items(items, k)
    start_time = time.perf_counter()
    rng = random.Random(seed)
    selected: set[T] = set()
    eval_count = 0
    current_value = objective(set())
    eval_count += 1

    if k == 0:
        return _build_result(selected, current_value, eval_count, start_time)

    sample_size = max(1, math.ceil((len(items_tuple) / k) * math.log(1.0 / epsilon)))

    for _ in range(k):
        remaining = [item for item in items_tuple if item not in selected]
        if not remaining:
            break

        actual_sample_size = min(sample_size, len(remaining))
        sampled_items = set(rng.sample(remaining, actual_sample_size))

        best_item: T | None = None
        best_gain = float("-inf")
        best_value = current_value

        for item in remaining:
            if item not in sampled_items:
                continue

            candidate_value = objective(selected | {item})
            eval_count += 1
            gain = candidate_value - current_value

            if gain > best_gain:
                best_item = item
                best_gain = gain
                best_value = candidate_value

        if best_item is None:
            break

        selected.add(best_item)
        current_value = best_value

    return _build_result(selected, current_value, eval_count, start_time)
