"""Run the small Facility Location experiment from ``SPEC.md``.

The script evaluates:

- brute force
- greedy
- random baseline

It exports CSV and LaTeX tables, and saves a scatter plot highlighting the
greedy representatives.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import brute_force, greedy, random_baseline
from src.facility_location import (
    facility_marginal_gain,
    facility_objective,
    gaussian_similarity,
    generate_cluster_data,
)


OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CSV_PATH = OUTPUT_TABLES_DIR / "facility_location_small.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "facility_location_small.tex"
FIGURE_PATH = OUTPUT_FIGURES_DIR / "facility_location_small.png"
K = 2
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def format_selected(selected: set[int]) -> str:
    """Return a stable string representation for selected indices."""

    return "{" + ", ".join(str(index) for index in sorted(selected)) + "}"


def dataframe_to_latex(table: pd.DataFrame) -> str:
    """Render a simple LaTeX tabular without optional pandas dependencies."""

    columns = list(table.columns)
    lines = [
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]

    for row in table.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_results_table() -> tuple[pd.DataFrame, set[int], FloatArray, IntArray]:
    """Run the small Facility Location experiment and return the summary table."""

    X, labels = generate_cluster_data(n_per_cluster=5, scale=0.4, seed=42)
    W = gaussian_similarity(X, sigma=1.0)
    items = list(range(len(X)))

    objective = lambda selected: facility_objective(W, selected)
    marginal_gain = lambda selected, x: facility_marginal_gain(W, x, selected)

    brute_force_result = brute_force(items, K, objective)
    greedy_result = greedy(items, K, objective, marginal_gain=marginal_gain)
    random_result = random_baseline(items, K, objective, n_trials=1000, seed=42)

    results = [
        ("Brute Force", brute_force_result),
        ("Greedy", greedy_result),
        ("Random Baseline", random_result),
    ]

    optimal_value = float(brute_force_result["value"])
    rows: list[dict[str, str | float | int]] = []
    for algorithm_name, result in results:
        value = float(result["value"])
        ratio = value / optimal_value if optimal_value > 0 else 0.0
        rows.append(
            {
                "Algorithm": algorithm_name,
                "Selected": format_selected(result["selected"]),
                "Objective Value": value,
                "Ratio to Optimal": ratio,
                "Evaluations": int(result["eval_count"]),
                "Runtime Seconds": float(result["runtime"]),
            }
        )

    table = pd.DataFrame(rows)
    return table, set(greedy_result["selected"]), X, labels


def save_outputs(table: pd.DataFrame) -> None:
    """Write CSV and LaTeX outputs to the tables directory."""

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False)
    LATEX_PATH.write_text(dataframe_to_latex(table), encoding="utf-8")


def save_scatter_plot(X: FloatArray, labels: IntArray, selected: set[int]) -> None:
    """Save a scatter plot with greedy-selected representatives highlighted."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="tab10",
        s=60,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        label="Data points",
    )

    selected_indices = sorted(selected)
    ax.scatter(
        X[selected_indices, 0],
        X[selected_indices, 1],
        marker="X",
        s=220,
        c="red",
        edgecolors="black",
        linewidths=1.0,
        label="Greedy representatives",
    )

    ax.set_title("Facility Location Small Instance")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def print_summary(table: pd.DataFrame) -> None:
    """Print a concise experiment summary to the terminal."""

    print("Facility Location small experiment")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")


def main() -> None:
    """Run the experiment and export results."""

    table, greedy_selected, X, labels = build_results_table()
    save_outputs(table)
    save_scatter_plot(X, labels, greedy_selected)
    print_summary(table)


if __name__ == "__main__":
    main()
