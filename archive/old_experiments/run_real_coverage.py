"""Run the real-world OSM Maximum Coverage experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import greedy, lazy_greedy, random_baseline, stochastic_greedy  # noqa: E402
from src.geo_coverage import (  # noqa: E402
    build_coverage_sets,
    coverage_marginal_gain_geo,
    coverage_objective_geo,
    covered_demand_indices,
    weighted_coverage_marginal_gain_geo,
    weighted_coverage_objective_geo,
)
from src.geo_metrics import (  # noqa: E402
    average_nearest_distance,
    max_nearest_distance,
    pairwise_distance_matrix,
)
from src.osm_data import load_processed_points  # noqa: E402
from src.osm_data import assign_demand_weights, is_priority_source_type  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CSV_PATH = OUTPUT_TABLES_DIR / "real_coverage_results.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "real_coverage_results.tex"
FIGURE_PATH = OUTPUT_FIGURES_DIR / "real_coverage_result.png"
RANDOM_TRIALS = 1000
EPSILON = 0.1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run real OSM Maximum Coverage.")
    parser.add_argument("--k", type=int, default=10, help="Number of facilities to select.")
    parser.add_argument(
        "--radius",
        type=float,
        default=300.0,
        help="Coverage radius in meters.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Deprecated alias for --weight-scheme priority_mild.",
    )
    parser.add_argument(
        "--weight-scheme",
        choices=["unweighted", "priority_mild", "priority_strong"],
        default="unweighted",
        help="Demand weighting scheme for the coverage objective.",
    )
    return parser.parse_args()


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
                values.append(f"{value:.6f}")
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


def apply_weight_scheme(
    demand_points: pd.DataFrame,
    weight_scheme: str,
) -> pd.DataFrame:
    """Assign demand weights for a selected scheme."""

    return assign_demand_weights(demand_points, weight_scheme)


def priority_mask_from_demand_table(demand_points: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask for priority demand points."""

    return demand_points["source_type"].map(is_priority_source_type).to_numpy(dtype=bool)


def build_metrics_row(
    algorithm_name: str,
    result: dict[str, object],
    coverage_sets: dict[int, set[int]],
    distance_matrix: np.ndarray,
    n_demand: int,
    demand_weights: np.ndarray,
    priority_mask: np.ndarray,
    weight_scheme: str,
) -> dict[str, float | int | str]:
    """Build one result-table row with coverage and distance metrics."""

    selected = set(int(index) for index in result["selected"])
    covered = covered_demand_indices(coverage_sets, selected)
    coverage_count = coverage_objective_geo(coverage_sets, selected, n_demand)
    coverage_rate = coverage_count / n_demand if n_demand > 0 else 0.0
    weighted_coverage_value = weighted_coverage_objective_geo(
        coverage_sets,
        selected,
        demand_weights,
    )
    total_weight = float(np.sum(demand_weights))
    weighted_coverage_rate = (
        weighted_coverage_value / total_weight if total_weight > 0 else 0.0
    )
    priority_points_total = int(np.sum(priority_mask))
    priority_points_covered = (
        int(np.sum(priority_mask[list(covered)])) if covered else 0
    )
    priority_coverage_rate = (
        priority_points_covered / priority_points_total
        if priority_points_total > 0
        else 0.0
    )

    if selected:
        average_distance = average_nearest_distance(distance_matrix, selected)
        max_distance = max_nearest_distance(distance_matrix, selected)
    else:
        average_distance = float("nan")
        max_distance = float("nan")

    return {
        "Algorithm": algorithm_name,
        "weight_scheme": weight_scheme,
        "objective_value": float(result["value"]),
        "coverage_count": int(coverage_count),
        "coverage_rate": float(coverage_rate),
        "total_weight": float(total_weight),
        "weighted_coverage_value": float(weighted_coverage_value),
        "weighted_coverage_rate": float(weighted_coverage_rate),
        "priority_points_total": priority_points_total,
        "priority_points_covered": priority_points_covered,
        "priority_coverage_rate": float(priority_coverage_rate),
        "average_nearest_distance_m": float(average_distance),
        "max_nearest_distance_m": float(max_distance),
        "eval_count": int(result["eval_count"]),
        "runtime_seconds": float(result["runtime"]),
    }


def run_algorithms(
    k: int,
    seed: int,
    coverage_sets: dict[int, set[int]],
    n_demand: int,
    n_candidates: int,
    demand_weights: np.ndarray,
    weight_scheme: str,
) -> list[tuple[str, dict[str, object]]]:
    """Run coverage algorithms on the real OSM coverage instance."""

    if k < 0:
        raise ValueError("k must be non-negative.")
    if k > n_candidates:
        raise ValueError(f"k={k} is larger than the number of candidates ({n_candidates}).")

    items = list(range(n_candidates))
    if weight_scheme != "unweighted":
        objective = lambda selected: weighted_coverage_objective_geo(
            coverage_sets,
            selected,
            demand_weights,
        )
        geo_marginal_gain = lambda x, selected: weighted_coverage_marginal_gain_geo(
            coverage_sets,
            x,
            selected,
            demand_weights,
        )
    else:
        objective = lambda selected: coverage_objective_geo(
            coverage_sets,
            selected,
            n_demand,
        )
        geo_marginal_gain = lambda x, selected: coverage_marginal_gain_geo(
            coverage_sets,
            x,
            selected,
        )
    marginal_gain = lambda selected, x: geo_marginal_gain(x, selected)

    return [
        ("Greedy", greedy(items, k, objective, marginal_gain=marginal_gain)),
        ("Lazy Greedy", lazy_greedy(items, k, objective)),
        (
            "Stochastic Greedy",
            stochastic_greedy(items, k, objective, epsilon=EPSILON, seed=seed),
        ),
        (
            "Random Baseline",
            random_baseline(items, k, objective, n_trials=RANDOM_TRIALS, seed=seed),
        ),
    ]


def save_outputs(table: pd.DataFrame) -> None:
    """Save result CSV and LaTeX tables."""

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False)
    LATEX_PATH.write_text(dataframe_to_latex(table), encoding="utf-8")


def save_coverage_plot(
    demand_points: pd.DataFrame,
    candidate_points: pd.DataFrame,
    coverage_sets: dict[int, set[int]],
    selected: set[int],
    radius: float,
) -> None:
    """Save the real coverage result figure."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    covered = covered_demand_indices(coverage_sets, selected)
    demand_indices = set(range(len(demand_points)))
    uncovered = demand_indices - covered

    fig, ax = plt.subplots(figsize=(8, 8))

    if uncovered:
        uncovered_rows = demand_points.iloc[sorted(uncovered)]
        ax.scatter(
            uncovered_rows["x"],
            uncovered_rows["y"],
            s=18,
            c="#d95f02",
            alpha=0.8,
            label="Uncovered demand",
        )

    if covered:
        covered_rows = demand_points.iloc[sorted(covered)]
        ax.scatter(
            covered_rows["x"],
            covered_rows["y"],
            s=18,
            c="#1b9e77",
            alpha=0.8,
            label="Covered demand",
        )

    ax.scatter(
        candidate_points["x"],
        candidate_points["y"],
        s=28,
        c="#7570b3",
        marker=".",
        alpha=0.55,
        label="Candidate locations",
    )

    if selected:
        selected_rows = candidate_points.iloc[sorted(selected)]
        ax.scatter(
            selected_rows["x"],
            selected_rows["y"],
            s=170,
            c="#111111",
            marker="*",
            edgecolors="white",
            linewidths=0.8,
            label="Selected facilities",
            zorder=5,
        )
        for row in selected_rows.itertuples(index=False):
            circle = Circle(
                (float(row.x), float(row.y)),
                radius,
                fill=False,
                edgecolor="#111111",
                linewidth=0.8,
                alpha=0.35,
            )
            ax.add_patch(circle)

    ax.set_title(f"Real OSM Maximum Coverage (radius = {radius:g} m)")
    ax.set_xlabel("Projected x (m)")
    ax.set_ylabel("Projected y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> int:
    """Run the real OSM Maximum Coverage experiment."""

    args = parse_args()

    try:
        demand_points, candidate_points = load_data()
        weight_scheme = args.weight_scheme
        if args.weighted and weight_scheme == "unweighted":
            weight_scheme = "priority_mild"
        demand_points = apply_weight_scheme(demand_points, weight_scheme)
        demand_xy = xy_from_table(demand_points)
        candidate_xy = xy_from_table(candidate_points)
        demand_weights = demand_points["weight"].to_numpy(dtype=np.float64)
        priority_mask = priority_mask_from_demand_table(demand_points)
        distance_matrix = pairwise_distance_matrix(demand_xy, candidate_xy)
        coverage_sets = build_coverage_sets(distance_matrix, args.radius)
        algorithm_results = run_algorithms(
            args.k,
            args.seed,
            coverage_sets,
            len(demand_points),
            len(candidate_points),
            demand_weights,
            weight_scheme,
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        print(exc, file=sys.stderr)
        return 1

    rows = [
        build_metrics_row(
            algorithm_name,
            result,
            coverage_sets,
            distance_matrix,
            len(demand_points),
            demand_weights,
            priority_mask,
            weight_scheme,
        )
        for algorithm_name, result in algorithm_results
    ]
    table = pd.DataFrame(rows)
    save_outputs(table)

    greedy_selected = set(int(index) for index in algorithm_results[0][1]["selected"])
    save_coverage_plot(
        demand_points,
        candidate_points,
        coverage_sets,
        greedy_selected,
        args.radius,
    )

    print(f"Real OSM Maximum Coverage experiment ({weight_scheme})")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
