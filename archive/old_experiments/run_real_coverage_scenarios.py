"""Run real-world OSM Maximum Coverage scenarios across radii and budgets."""

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

from src.algorithms import greedy, lazy_greedy, random_baseline, stochastic_greedy  # noqa: E402
from src.geo_coverage import (  # noqa: E402
    build_coverage_sets,
    coverage_marginal_gain_geo,
    coverage_objective_geo,
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
CSV_PATH = OUTPUT_TABLES_DIR / "real_coverage_scenarios.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "real_coverage_scenarios.tex"
COVERAGE_BY_K_PATH = OUTPUT_FIGURES_DIR / "coverage_rate_by_k.png"
COVERAGE_BY_RADIUS_PATH = OUTPUT_FIGURES_DIR / "coverage_rate_by_radius.png"

RADII = [100.0, 150.0, 200.0]
K_VALUES = [5, 10, 15, 20, 30]
EPSILON = 0.1
RANDOM_TRIALS = 1000
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
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append(" & ".join(values) + " \\\\")

    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def xy_from_table(table: pd.DataFrame) -> np.ndarray:
    """Extract projected x/y coordinates from a processed points table."""

    return table[["x", "y"]].to_numpy(dtype=np.float64)


def run_algorithms(
    items: list[int],
    k: int,
    coverage_sets: dict[int, set[int]],
    n_demand: int,
) -> list[tuple[str, dict[str, object]]]:
    """Run the four coverage algorithms for one scenario."""

    objective = lambda selected: coverage_objective_geo(coverage_sets, selected, n_demand)
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
            stochastic_greedy(items, k, objective, epsilon=EPSILON, seed=SEED),
        ),
        (
            "Random Baseline",
            random_baseline(items, k, objective, n_trials=RANDOM_TRIALS, seed=SEED),
        ),
    ]


def build_rows(
    distance_matrix: np.ndarray,
    n_demand: int,
    n_candidates: int,
) -> list[dict[str, float | int | str]]:
    """Build all scenario result rows."""

    rows: list[dict[str, float | int | str]] = []
    items = list(range(n_candidates))

    for radius in RADII:
        coverage_sets = build_coverage_sets(distance_matrix, radius)
        for k in K_VALUES:
            if k > n_candidates:
                continue
            for algorithm_name, result in run_algorithms(items, k, coverage_sets, n_demand):
                selected = set(int(index) for index in result["selected"])
                coverage_count = int(result["value"])
                coverage_rate = coverage_count / n_demand if n_demand > 0 else 0.0
                rows.append(
                    {
                        "radius_m": int(radius),
                        "k": k,
                        "algorithm": algorithm_name,
                        "coverage_count": coverage_count,
                        "coverage_rate": float(coverage_rate),
                        "avg_nearest_distance_m": average_nearest_distance(
                            distance_matrix,
                            selected,
                        ),
                        "max_nearest_distance_m": max_nearest_distance(
                            distance_matrix,
                            selected,
                        ),
                        "eval_count": int(result["eval_count"]),
                        "runtime_seconds": float(result["runtime"]),
                    }
                )

    return rows


def save_outputs(table: pd.DataFrame) -> None:
    """Save scenario CSV and LaTeX tables."""

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False)
    LATEX_PATH.write_text(dataframe_to_latex(table), encoding="utf-8")


def save_coverage_rate_by_k(table: pd.DataFrame) -> None:
    """Plot coverage rate by k, faceted by radius."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(RADII), figsize=(14, 4.6), sharey=True)

    for ax, radius in zip(axes, RADII, strict=True):
        subset = table[table["radius_m"] == int(radius)]
        for algorithm_name in subset["algorithm"].unique():
            algorithm_rows = subset[subset["algorithm"] == algorithm_name].sort_values("k")
            ax.plot(
                algorithm_rows["k"],
                algorithm_rows["coverage_rate"],
                marker="o",
                linewidth=2,
                label=algorithm_name,
            )
        ax.set_title(f"Radius {int(radius)} m")
        ax.set_xlabel("k")
        ax.set_ylabel("Coverage rate")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(COVERAGE_BY_K_PATH, dpi=200)
    plt.close(fig)


def save_coverage_rate_by_radius(table: pd.DataFrame) -> None:
    """Plot coverage rate by radius, faceted by k."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(17, 4.4), sharey=True)

    for ax, k in zip(axes, K_VALUES, strict=True):
        subset = table[table["k"] == k]
        for algorithm_name in subset["algorithm"].unique():
            algorithm_rows = subset[subset["algorithm"] == algorithm_name].sort_values(
                "radius_m"
            )
            ax.plot(
                algorithm_rows["radius_m"],
                algorithm_rows["coverage_rate"],
                marker="o",
                linewidth=2,
                label=algorithm_name,
            )
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Radius (m)")
        ax.set_ylabel("Coverage rate")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(COVERAGE_BY_RADIUS_PATH, dpi=200)
    plt.close(fig)


def save_plots(table: pd.DataFrame) -> None:
    """Save all scenario figures."""

    save_coverage_rate_by_k(table)
    save_coverage_rate_by_radius(table)


def main() -> int:
    """Run all real OSM Maximum Coverage scenarios."""

    try:
        demand_points, candidate_points = load_processed_points(DATA_DIR)
        demand_xy = xy_from_table(demand_points)
        candidate_xy = xy_from_table(candidate_points)
        distance_matrix = pairwise_distance_matrix(demand_xy, candidate_xy)
    except (FileNotFoundError, ValueError) as exc:
        print(
            "Processed OSM data not found or invalid. Run "
            "`python experiments/run_osm_data_collection.py` first. "
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    rows = build_rows(distance_matrix, len(demand_points), len(candidate_points))
    table = pd.DataFrame(rows)
    save_outputs(table)
    save_plots(table)

    print("Real OSM Maximum Coverage scenarios")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")
    print(f"Coverage by k figure saved to: {COVERAGE_BY_K_PATH}")
    print(f"Coverage by radius figure saved to: {COVERAGE_BY_RADIUS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
