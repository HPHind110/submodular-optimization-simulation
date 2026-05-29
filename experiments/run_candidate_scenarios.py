"""Compare bus-stop-only and road-node candidate sets for real coverage."""

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

from src.algorithms import greedy, lazy_greedy  # noqa: E402
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
from src.osm_data import DEFAULT_PLACE, collect_osm_points, load_processed_points  # noqa: E402


DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEMAND_PATH = DATA_DIR / "demand_points.csv"
BUS_STOP_CANDIDATE_PATH = DATA_DIR / "candidate_points_bus_stop_only.csv"
ROAD_NODE_CANDIDATE_PATH = DATA_DIR / "candidate_points_road_nodes.csv"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CSV_PATH = OUTPUT_TABLES_DIR / "candidate_scenario_comparison.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "candidate_scenario_comparison.tex"
FIGURE_PATH = OUTPUT_FIGURES_DIR / "candidate_scenario_coverage.png"

MAX_CANDIDATES = 400
MAX_ROAD_NODE_CANDIDATES = 400
NETWORK_TYPE = "walk"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Compare OSM candidate scenarios.")
    parser.add_argument(
        "--radius",
        type=float,
        default=150.0,
        help="Coverage radius in meters.",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20],
        help="Cardinality budgets to evaluate.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def radius_label(radius: float) -> str:
    """Return a compact radius label for output filenames."""

    return str(int(radius)) if float(radius).is_integer() else str(radius).replace(".", "p")


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


def ensure_candidate_csvs(seed: int) -> None:
    """Create scenario candidate CSV files if they are missing."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not BUS_STOP_CANDIDATE_PATH.exists():
        _, bus_stop_candidates = collect_osm_points(
            place_name=DEFAULT_PLACE,
            max_candidates=MAX_CANDIDATES,
            seed=seed,
            include_road_nodes=False,
        )
        bus_stop_candidates.to_csv(BUS_STOP_CANDIDATE_PATH, index=False)

    if not ROAD_NODE_CANDIDATE_PATH.exists():
        _, road_node_candidates = collect_osm_points(
            place_name=DEFAULT_PLACE,
            max_candidates=MAX_CANDIDATES,
            seed=seed,
            include_road_nodes=True,
            max_road_node_candidates=MAX_ROAD_NODE_CANDIDATES,
            network_type=NETWORK_TYPE,
        )
        road_node_candidates.to_csv(ROAD_NODE_CANDIDATE_PATH, index=False)


def load_demand_points() -> pd.DataFrame:
    """Load the shared demand points table."""

    if not DEMAND_PATH.exists():
        raise FileNotFoundError(
            "Processed demand_points.csv not found. Run "
            "`python experiments/run_osm_data_collection.py` first."
        )
    demand_points, _ = load_processed_points(DATA_DIR)
    return demand_points


def run_algorithms(
    items: list[int],
    k: int,
    coverage_sets: dict[int, set[int]],
    n_demand: int,
) -> list[tuple[str, dict[str, object]]]:
    """Run Greedy and Lazy Greedy for one scenario."""

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
    ]


def scenario_rows(
    scenario: str,
    demand_points: pd.DataFrame,
    candidate_points: pd.DataFrame,
    radius: float,
    k_values: list[int],
) -> list[dict[str, float | int | str]]:
    """Build comparison rows for one candidate scenario."""

    distance_matrix = pairwise_distance_matrix(
        xy_from_table(demand_points),
        xy_from_table(candidate_points),
    )
    coverage_sets = build_coverage_sets(distance_matrix, radius)
    n_demand = len(demand_points)
    n_candidate = len(candidate_points)
    items = list(range(n_candidate))
    rows: list[dict[str, float | int | str]] = []

    for k in k_values:
        if k > n_candidate:
            continue
        for algorithm_name, result in run_algorithms(items, k, coverage_sets, n_demand):
            selected = set(int(index) for index in result["selected"])
            coverage_count = int(result["value"])
            rows.append(
                {
                    "scenario": scenario,
                    "n_demand": n_demand,
                    "n_candidate": n_candidate,
                    "radius_m": int(radius),
                    "k": k,
                    "algorithm": algorithm_name,
                    "coverage_count": coverage_count,
                    "coverage_rate": coverage_count / n_demand if n_demand > 0 else 0.0,
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


def save_outputs(table: pd.DataFrame, radius: float) -> tuple[Path, Path]:
    """Save comparison CSV and LaTeX tables."""

    label = radius_label(radius)
    csv_path = OUTPUT_TABLES_DIR / f"candidate_scenario_comparison_R{label}.csv"
    latex_path = OUTPUT_TABLES_DIR / f"candidate_scenario_comparison_R{label}.tex"
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    latex_path.write_text(dataframe_to_latex(table), encoding="utf-8")
    return csv_path, latex_path


def save_coverage_plot(table: pd.DataFrame, radius: float) -> Path:
    """Save coverage-rate comparison figure."""

    label = radius_label(radius)
    figure_path = OUTPUT_FIGURES_DIR / f"candidate_scenario_coverage_R{label}.png"
    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))

    for scenario in table["scenario"].unique():
        scenario_rows_table = table[table["scenario"] == scenario]
        for algorithm_name in scenario_rows_table["algorithm"].unique():
            subset = scenario_rows_table[
                scenario_rows_table["algorithm"] == algorithm_name
            ].sort_values("k")
            ax.plot(
                subset["k"],
                subset["coverage_rate"],
                marker="o",
                linewidth=2,
                label=f"{scenario} - {algorithm_name}",
            )

    ax.set_title("Candidate Scenario Coverage Comparison")
    ax.set_xlabel("k")
    ax.set_ylabel("Coverage rate")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def main() -> int:
    """Run candidate set scenario comparison."""

    args = parse_args()

    try:
        demand_points = load_demand_points()
        ensure_candidate_csvs(args.seed)
        bus_stop_candidates = pd.read_csv(BUS_STOP_CANDIDATE_PATH)
        road_node_candidates = pd.read_csv(ROAD_NODE_CANDIDATE_PATH)
    except Exception as exc:
        print(f"Could not load or create candidate scenarios. Error: {exc}", file=sys.stderr)
        return 1

    rows = scenario_rows(
        "bus_stop_only",
        demand_points,
        bus_stop_candidates,
        args.radius,
        args.ks,
    )
    rows.extend(
        scenario_rows(
            "road_nodes",
            demand_points,
            road_node_candidates,
            args.radius,
            args.ks,
        )
    )

    table = pd.DataFrame(rows)
    csv_path, latex_path = save_outputs(table, args.radius)
    figure_path = save_coverage_plot(table, args.radius)

    print("Candidate scenario comparison")
    print(table.to_string(index=False))
    print(f"Bus-stop-only candidate CSV: {BUS_STOP_CANDIDATE_PATH}")
    print(f"Road-node candidate CSV: {ROAD_NODE_CANDIDATE_PATH}")
    print(f"CSV saved to: {csv_path}")
    print(f"LaTeX saved to: {latex_path}")
    print(f"Figure saved to: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
