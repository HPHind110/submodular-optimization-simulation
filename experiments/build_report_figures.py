"""Build final Chapter 4 figures from validated experiment outputs."""

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

from src.algorithms import greedy  # noqa: E402
from src.geo_coverage import (  # noqa: E402
    build_coverage_sets,
    coverage_marginal_gain_geo,
    coverage_objective_geo,
    covered_demand_indices,
)
from src.geo_metrics import pairwise_distance_matrix  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RADII = [100, 150, 200]


def xy_from_table(table: pd.DataFrame) -> np.ndarray:
    """Extract projected x/y coordinates from a processed points table."""

    return table[["x", "y"]].to_numpy(dtype=np.float64)


def load_candidate_tables() -> pd.DataFrame:
    """Load candidate comparison outputs for all radii."""

    frames = []
    for radius in RADII:
        path = OUTPUT_TABLES_DIR / f"candidate_scenario_comparison_R{radius}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate scenario table: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def save_coverage_rate_by_k_all_radii(candidate_table: pd.DataFrame) -> Path:
    """Save coverage rate by k for both scenarios and all radii."""

    path = OUTPUT_FIGURES_DIR / "report_coverage_rate_by_k_all_radii.png"
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(RADII), figsize=(14, 4.6), sharey=True)

    for ax, radius in zip(axes, RADII, strict=True):
        subset = candidate_table[
            (candidate_table["radius_m"] == radius)
            & (candidate_table["algorithm"] == "Greedy")
        ]
        for scenario in subset["scenario"].unique():
            scenario_rows = subset[subset["scenario"] == scenario].sort_values("k")
            ax.plot(
                scenario_rows["k"],
                scenario_rows["coverage_rate"],
                marker="o",
                linewidth=2,
                label=scenario,
            )
        ax.set_title(f"R = {radius} m")
        ax.set_xlabel("k")
        ax.set_ylabel("Coverage rate")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_lazy_eval_reduction(candidate_table: pd.DataFrame) -> Path:
    """Save Lazy Greedy evaluation reduction at R=150."""

    path = OUTPUT_FIGURES_DIR / "report_lazy_eval_reduction_R150.png"
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    radius_table = candidate_table[candidate_table["radius_m"] == 150]
    rows: list[dict[str, float | int | str]] = []
    for (scenario, k), subset in radius_table.groupby(["scenario", "k"], sort=False):
        greedy_row = subset[subset["algorithm"] == "Greedy"].iloc[0]
        lazy_row = subset[subset["algorithm"] == "Lazy Greedy"].iloc[0]
        reduction = 1.0 - (int(lazy_row["eval_count"]) / int(greedy_row["eval_count"]))
        rows.append({"scenario": scenario, "k": int(k), "eval_reduction_rate": reduction})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for scenario in table["scenario"].unique():
        subset = table[table["scenario"] == scenario].sort_values("k")
        ax.plot(
            subset["k"],
            subset["eval_reduction_rate"],
            marker="o",
            linewidth=2,
            label=scenario,
        )
    ax.set_title("Lazy Greedy Evaluation Reduction (R = 150 m)")
    ax.set_xlabel("k")
    ax.set_ylabel("Evaluation reduction rate")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def selected_for_greedy(
    demand_points: pd.DataFrame,
    candidate_points: pd.DataFrame,
    radius: float,
    k: int,
) -> tuple[set[int], set[int]]:
    """Run Greedy and return selected candidates and covered demand indices."""

    distance_matrix = pairwise_distance_matrix(
        xy_from_table(demand_points),
        xy_from_table(candidate_points),
    )
    coverage_sets = build_coverage_sets(distance_matrix, radius)
    objective = lambda selected: coverage_objective_geo(
        coverage_sets,
        selected,
        len(demand_points),
    )
    geo_marginal_gain = lambda x, selected: coverage_marginal_gain_geo(
        coverage_sets,
        x,
        selected,
    )
    result = greedy(
        list(range(len(candidate_points))),
        k,
        objective,
        marginal_gain=lambda selected, x: geo_marginal_gain(x, selected),
    )
    selected = set(int(index) for index in result["selected"])
    return selected, covered_demand_indices(coverage_sets, selected)


def save_candidate_scenario_map() -> Path:
    """Save a map-like projected scatter for R=150, k=10."""

    path = OUTPUT_FIGURES_DIR / "report_candidate_scenario_map_R150_k10.png"
    demand_points = pd.read_csv(DATA_DIR / "demand_points.csv")
    bus_candidates = pd.read_csv(DATA_DIR / "candidate_points_bus_stop_only.csv")
    road_candidates = pd.read_csv(DATA_DIR / "candidate_points_road_nodes.csv")
    bus_selected, bus_covered = selected_for_greedy(demand_points, bus_candidates, 150.0, 10)
    road_selected, road_covered = selected_for_greedy(demand_points, road_candidates, 150.0, 10)

    fig, ax = plt.subplots(figsize=(8, 8))
    demand_index = set(range(len(demand_points)))
    uncovered_by_road = demand_index - road_covered
    if uncovered_by_road:
        rows = demand_points.iloc[sorted(uncovered_by_road)]
        ax.scatter(rows["x"], rows["y"], s=10, c="#d95f02", alpha=0.45, label="Uncovered demand")
    if road_covered:
        rows = demand_points.iloc[sorted(road_covered)]
        ax.scatter(rows["x"], rows["y"], s=10, c="#1b9e77", alpha=0.45, label="Covered demand")

    ax.scatter(
        bus_candidates.iloc[sorted(bus_selected)]["x"],
        bus_candidates.iloc[sorted(bus_selected)]["y"],
        s=120,
        c="#7570b3",
        marker="s",
        edgecolors="white",
        linewidths=0.7,
        label="Selected bus-stop candidates",
        zorder=5,
    )
    ax.scatter(
        road_candidates.iloc[sorted(road_selected)]["x"],
        road_candidates.iloc[sorted(road_selected)]["y"],
        s=150,
        c="#111111",
        marker="*",
        edgecolors="white",
        linewidths=0.7,
        label="Selected road-node candidates",
        zorder=6,
    )
    ax.set_title("Candidate Scenario Map (R = 150 m, k = 10)")
    ax.set_xlabel("Projected x (m)")
    ax.set_ylabel("Projected y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> int:
    """Build all final report figures."""

    try:
        candidate_table = load_candidate_tables()
        coverage_path = save_coverage_rate_by_k_all_radii(candidate_table)
        lazy_path = save_lazy_eval_reduction(candidate_table)
        map_path = save_candidate_scenario_map()
    except Exception as exc:
        print(f"Could not build report figures. Error: {exc}", file=sys.stderr)
        return 1

    print("Report figures built")
    print(f"Coverage figure: {coverage_path}")
    print(f"Lazy efficiency figure: {lazy_path}")
    print(f"Candidate scenario map: {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
