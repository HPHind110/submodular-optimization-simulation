"""Run the real-world OSM Facility Location experiment."""

from __future__ import annotations

import argparse
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

from src.algorithms import greedy, lazy_greedy, random_baseline, stochastic_greedy  # noqa: E402
from src.geo_facility_location import (  # noqa: E402
    build_similarity_matrix,
    facility_marginal_gain_geo,
    facility_objective_geo,
    selected_assignment,
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
CSV_PATH = OUTPUT_TABLES_DIR / "real_facility_results.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "real_facility_results.tex"
FIGURE_PATH = OUTPUT_FIGURES_DIR / "real_facility_result.png"
RANDOM_TRIALS = 1000
EPSILON = 0.1
MAX_ASSIGNMENT_LINES = 300


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run real OSM Facility Location.")
    parser.add_argument("--k", type=int, default=10, help="Number of facilities to select.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=300.0,
        help="Gaussian similarity bandwidth in meters.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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


def build_metrics_row(
    algorithm_name: str,
    result: dict[str, object],
    distance_matrix: np.ndarray,
) -> dict[str, float | int | str]:
    """Build one result-table row with objective and distance metrics."""

    selected = set(int(index) for index in result["selected"])
    if selected:
        average_distance = average_nearest_distance(distance_matrix, selected)
        max_distance = max_nearest_distance(distance_matrix, selected)
    else:
        average_distance = float("nan")
        max_distance = float("nan")

    return {
        "Algorithm": algorithm_name,
        "objective_value": float(result["value"]),
        "average_nearest_distance_m": float(average_distance),
        "max_nearest_distance_m": float(max_distance),
        "eval_count": int(result["eval_count"]),
        "runtime_seconds": float(result["runtime"]),
    }


def run_algorithms(
    k: int,
    seed: int,
    W: np.ndarray,
    n_candidates: int,
) -> list[tuple[str, dict[str, object]]]:
    """Run Facility Location algorithms on the real OSM instance."""

    if k < 0:
        raise ValueError("k must be non-negative.")
    if k > n_candidates:
        raise ValueError(f"k={k} is larger than the number of candidates ({n_candidates}).")

    items = list(range(n_candidates))
    objective = lambda selected: facility_objective_geo(W, selected)
    geo_marginal_gain = lambda x, selected: facility_marginal_gain_geo(W, x, selected)
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


def save_facility_plot(
    demand_points: pd.DataFrame,
    candidate_points: pd.DataFrame,
    distance_matrix: np.ndarray,
    selected: set[int],
) -> None:
    """Save the real Facility Location result figure."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        demand_points["x"],
        demand_points["y"],
        s=18,
        c="#1b9e77",
        alpha=0.75,
        label="Demand points",
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
        assignments = selected_assignment(distance_matrix, selected)

        if len(demand_points) <= MAX_ASSIGNMENT_LINES:
            for demand_index, candidate_index in enumerate(assignments):
                demand_row = demand_points.iloc[demand_index]
                candidate_row = candidate_points.iloc[int(candidate_index)]
                ax.plot(
                    [demand_row["x"], candidate_row["x"]],
                    [demand_row["y"], candidate_row["y"]],
                    color="#666666",
                    linewidth=0.45,
                    alpha=0.25,
                    zorder=1,
                )

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

    ax.set_title("Real OSM Facility Location")
    ax.set_xlabel("Projected x (m)")
    ax.set_ylabel("Projected y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=200)
    plt.close(fig)


def main() -> int:
    """Run the real OSM Facility Location experiment."""

    args = parse_args()

    try:
        demand_points, candidate_points = load_data()
        demand_xy = xy_from_table(demand_points)
        candidate_xy = xy_from_table(candidate_points)
        distance_matrix = pairwise_distance_matrix(demand_xy, candidate_xy)
        W = build_similarity_matrix(distance_matrix, args.sigma)
        algorithm_results = run_algorithms(
            args.k,
            args.seed,
            W,
            len(candidate_points),
        )
    except (FileNotFoundError, ValueError, IndexError) as exc:
        print(exc, file=sys.stderr)
        return 1

    rows = [
        build_metrics_row(algorithm_name, result, distance_matrix)
        for algorithm_name, result in algorithm_results
    ]
    table = pd.DataFrame(rows)
    save_outputs(table)

    greedy_selected = set(int(index) for index in algorithm_results[0][1]["selected"])
    save_facility_plot(demand_points, candidate_points, distance_matrix, greedy_selected)

    print("Real OSM Facility Location experiment")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")
    print(f"Figure saved to: {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
