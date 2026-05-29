"""Compare runtime and evaluation counts on Facility Location.

This script generates two-cluster Gaussian data for several dataset sizes and
compares:

- greedy
- lazy greedy
- stochastic greedy

Outputs:

- ``outputs/tables/runtime_comparison.csv``
- ``outputs/tables/runtime_comparison.tex``
- ``outputs/figures/runtime_comparison.png``
- ``outputs/figures/evaluation_comparison.png``
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import greedy, lazy_greedy, stochastic_greedy
from src.facility_location import facility_objective, gaussian_similarity, generate_cluster_data


OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CSV_PATH = OUTPUT_TABLES_DIR / "runtime_comparison.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "runtime_comparison.tex"
RUNTIME_FIGURE_PATH = OUTPUT_FIGURES_DIR / "runtime_comparison.png"
EVALUATION_FIGURE_PATH = OUTPUT_FIGURES_DIR / "evaluation_comparison.png"

N_PER_CLUSTER_VALUES = [100, 250, 500]
SIGMA = 1.0
K = 10
EPSILON = 0.1
SEED = 42


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


def build_results_table() -> pd.DataFrame:
    """Run runtime experiments across dataset sizes and return the results."""

    rows: list[dict[str, int | float | str]] = []

    for n_per_cluster in N_PER_CLUSTER_VALUES:
        X, _ = generate_cluster_data(
            n_per_cluster=n_per_cluster,
            scale=0.4,
            seed=SEED,
        )
        W = gaussian_similarity(X, sigma=SIGMA)
        items = list(range(len(X)))
        objective = lambda selected, matrix=W: facility_objective(matrix, selected)

        algorithm_runs = [
            ("Greedy", greedy(items, K, objective)),
            ("Lazy Greedy", lazy_greedy(items, K, objective)),
            (
                "Stochastic Greedy",
                stochastic_greedy(items, K, objective, epsilon=EPSILON, seed=SEED),
            ),
        ]

        total_points = len(X)
        for algorithm_name, result in algorithm_runs:
            rows.append(
                {
                    "n": total_points,
                    "Algorithm": algorithm_name,
                    "Objective Value": float(result["value"]),
                    "Evaluations": int(result["eval_count"]),
                    "Runtime Seconds": float(result["runtime"]),
                }
            )

    return pd.DataFrame(rows)


def save_outputs(table: pd.DataFrame) -> None:
    """Write CSV and LaTeX tables to disk."""

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False)
    LATEX_PATH.write_text(dataframe_to_latex(table), encoding="utf-8")


def save_runtime_plot(table: pd.DataFrame) -> None:
    """Save the runtime comparison figure."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    for algorithm_name in table["Algorithm"].unique():
        subset = table[table["Algorithm"] == algorithm_name].sort_values("n")
        ax.plot(
            subset["n"],
            subset["Runtime Seconds"],
            marker="o",
            linewidth=2,
            label=algorithm_name,
        )

    ax.set_title("Facility Location Runtime Comparison")
    ax.set_xlabel("Number of data points (n)")
    ax.set_ylabel("Runtime (seconds)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(RUNTIME_FIGURE_PATH, dpi=200)
    plt.close(fig)


def save_evaluation_plot(table: pd.DataFrame) -> None:
    """Save the evaluation count comparison figure."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    for algorithm_name in table["Algorithm"].unique():
        subset = table[table["Algorithm"] == algorithm_name].sort_values("n")
        ax.plot(
            subset["n"],
            subset["Evaluations"],
            marker="o",
            linewidth=2,
            label=algorithm_name,
        )

    ax.set_title("Facility Location Evaluation Comparison")
    ax.set_xlabel("Number of data points (n)")
    ax.set_ylabel("Evaluations")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(EVALUATION_FIGURE_PATH, dpi=200)
    plt.close(fig)


def print_summary(table: pd.DataFrame) -> None:
    """Print a concise summary to the terminal."""

    print("Facility Location runtime comparison")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")
    print(f"Runtime figure saved to: {RUNTIME_FIGURE_PATH}")
    print(f"Evaluation figure saved to: {EVALUATION_FIGURE_PATH}")


def main() -> None:
    """Run the experiment suite and export tables and figures."""

    table = build_results_table()
    save_outputs(table)
    save_runtime_plot(table)
    save_evaluation_plot(table)
    print_summary(table)


if __name__ == "__main__":
    main()
