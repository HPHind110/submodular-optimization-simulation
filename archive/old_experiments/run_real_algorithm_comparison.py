"""Compare real-world OSM algorithms across different cardinality budgets."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import greedy, lazy_greedy, stochastic_greedy  # noqa: E402
from src.geo_coverage import (  # noqa: E402
    build_coverage_sets,
    coverage_marginal_gain_geo,
    coverage_objective_geo,
)
from src.geo_facility_location import (  # noqa: E402
    build_similarity_matrix,
    facility_marginal_gain_geo,
    facility_objective_geo,
)
from src.geo_metrics import (  # noqa: E402
    average_nearest_distance,
    max_nearest_distance,
    pairwise_distance_matrix,
)
from src.osm_data import load_processed_points  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CSV_PATH = OUTPUT_TABLES_DIR / "real_algorithm_comparison.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "real_algorithm_comparison.tex"
RUNTIME_FIGURE_PATH = OUTPUT_FIGURES_DIR / "real_runtime_comparison.png"
EVALUATION_FIGURE_PATH = OUTPUT_FIGURES_DIR / "real_evaluation_comparison.png"
OBJECTIVE_FIGURE_PATH = OUTPUT_FIGURES_DIR / "real_objective_comparison.png"

K_VALUES = [5, 10, 15, 20]
RADIUS = 300.0
SIGMA = 300.0
EPSILON = 0.1
SEED = 42


def dataframe_to_latex(table: pd.DataFrame) -> str:
    """Render a compact LaTeX tabular."""

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
                values.append(f"{value:.6f}" if np.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed OSM demand and candidate points."""

    try:
        return load_processed_points(DATA_DIR)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Processed OSM data not found. Run "
            "`python experiments/run_osm_data_collection.py` first."
        ) from exc


def xy_from_table(table: pd.DataFrame) -> np.ndarray:
    """Extract projected x/y coordinates from a processed points table."""

    return table[["x", "y"]].to_numpy(dtype=np.float64)


def algorithm_runs(
    items: list[int],
    k: int,
    objective,
    marginal_gain,
) -> list[tuple[str, dict[str, object]]]:
    """Run the three comparison algorithms."""

    return [
        ("Greedy", greedy(items, k, objective, marginal_gain=marginal_gain)),
        ("Lazy Greedy", lazy_greedy(items, k, objective)),
        (
            "Stochastic Greedy",
            stochastic_greedy(items, k, objective, epsilon=EPSILON, seed=SEED),
        ),
    ]


def coverage_rows(
    distance_matrix: np.ndarray,
    n_demand: int,
    n_candidates: int,
) -> list[dict[str, float | int | str]]:
    """Build comparison rows for Maximum Coverage."""

    coverage_sets = build_coverage_sets(distance_matrix, RADIUS)
    rows: list[dict[str, float | int | str]] = []
    items = list(range(n_candidates))

    for k in K_VALUES:
        if k > n_candidates:
            continue

        objective = lambda selected: coverage_objective_geo(coverage_sets, selected, n_demand)
        geo_marginal_gain = lambda x, selected: coverage_marginal_gain_geo(
            coverage_sets,
            x,
            selected,
        )
        marginal_gain = lambda selected, x: geo_marginal_gain(x, selected)

        for algorithm_name, result in algorithm_runs(items, k, objective, marginal_gain):
            selected = set(int(index) for index in result["selected"])
            coverage_count = int(result["value"])
            coverage_rate = coverage_count / n_demand if n_demand > 0 else 0.0
            rows.append(
                {
                    "problem": "Maximum Coverage",
                    "k": k,
                    "algorithm": algorithm_name,
                    "objective_value": float(result["value"]),
                    "coverage_rate": float(coverage_rate),
                    "avg_nearest_distance_m": average_nearest_distance(distance_matrix, selected),
                    "max_nearest_distance_m": max_nearest_distance(distance_matrix, selected),
                    "eval_count": int(result["eval_count"]),
                    "runtime_seconds": float(result["runtime"]),
                }
            )

    return rows


def facility_rows(
    distance_matrix: np.ndarray,
    n_candidates: int,
) -> list[dict[str, float | int | str]]:
    """Build comparison rows for Facility Location."""

    W = build_similarity_matrix(distance_matrix, SIGMA)
    rows: list[dict[str, float | int | str]] = []
    items = list(range(n_candidates))

    for k in K_VALUES:
        if k > n_candidates:
            continue

        objective = lambda selected: facility_objective_geo(W, selected)
        geo_marginal_gain = lambda x, selected: facility_marginal_gain_geo(W, x, selected)
        marginal_gain = lambda selected, x: geo_marginal_gain(x, selected)

        for algorithm_name, result in algorithm_runs(items, k, objective, marginal_gain):
            selected = set(int(index) for index in result["selected"])
            rows.append(
                {
                    "problem": "Facility Location",
                    "k": k,
                    "algorithm": algorithm_name,
                    "objective_value": float(result["value"]),
                    "coverage_rate": np.nan,
                    "avg_nearest_distance_m": average_nearest_distance(distance_matrix, selected),
                    "max_nearest_distance_m": max_nearest_distance(distance_matrix, selected),
                    "eval_count": int(result["eval_count"]),
                    "runtime_seconds": float(result["runtime"]),
                }
            )

    return rows


def save_outputs(table: pd.DataFrame) -> None:
    """Save comparison CSV and LaTeX tables."""

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False)
    LATEX_PATH.write_text(dataframe_to_latex(table), encoding="utf-8")


def save_metric_plot(
    table: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Save a line plot for one metric across k values."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True)

    for ax, problem in zip(axes, ["Facility Location", "Maximum Coverage"], strict=True):
        subset = table[table["problem"] == problem]
        for algorithm_name in subset["algorithm"].unique():
            algorithm_rows = subset[subset["algorithm"] == algorithm_name].sort_values("k")
            ax.plot(
                algorithm_rows["k"],
                algorithm_rows[metric],
                marker="o",
                linewidth=2,
                label=algorithm_name,
            )

        ax.set_title(problem)
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_plots(table: pd.DataFrame) -> None:
    """Save runtime, evaluation count, and objective comparison figures."""

    save_metric_plot(
        table,
        "runtime_seconds",
        "Runtime (seconds)",
        RUNTIME_FIGURE_PATH,
    )
    save_metric_plot(
        table,
        "eval_count",
        "Evaluations",
        EVALUATION_FIGURE_PATH,
    )
    save_metric_plot(
        table,
        "objective_value",
        "Objective value",
        OBJECTIVE_FIGURE_PATH,
    )


def main() -> int:
    """Run the real OSM algorithm comparison."""

    try:
        demand_points, candidate_points = load_data()
        demand_xy = xy_from_table(demand_points)
        candidate_xy = xy_from_table(candidate_points)
        distance_matrix = pairwise_distance_matrix(demand_xy, candidate_xy)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    n_demand = len(demand_points)
    n_candidates = len(candidate_points)
    rows = facility_rows(distance_matrix, n_candidates)
    rows.extend(coverage_rows(distance_matrix, n_demand, n_candidates))

    table = pd.DataFrame(rows)
    save_outputs(table)
    save_plots(table)

    print("Real OSM algorithm comparison")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")
    print(f"Runtime figure saved to: {RUNTIME_FIGURE_PATH}")
    print(f"Evaluation figure saved to: {EVALUATION_FIGURE_PATH}")
    print(f"Objective figure saved to: {OBJECTIVE_FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
