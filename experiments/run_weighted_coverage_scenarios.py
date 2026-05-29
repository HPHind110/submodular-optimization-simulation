"""Run weighted Maximum Coverage scenarios for real OSM demand priorities."""

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

from src.algorithms import greedy, lazy_greedy  # noqa: E402
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
from src.osm_data import (  # noqa: E402
    assign_demand_weights,
    is_priority_source_type,
    load_processed_points,
)


DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
CSV_PATH = OUTPUT_TABLES_DIR / "weighted_coverage_scenarios.csv"
LATEX_PATH = OUTPUT_TABLES_DIR / "weighted_coverage_scenarios.tex"
REPORT_CSV_PATH = OUTPUT_TABLES_DIR / "report_weighted_vs_unweighted_R150.csv"
REPORT_LATEX_PATH = OUTPUT_TABLES_DIR / "report_weighted_vs_unweighted_R150.tex"
WEIGHTED_VALUE_FIGURE_PATH = OUTPUT_FIGURES_DIR / "weighted_coverage_value_by_k.png"
PRIORITY_RATE_FIGURE_PATH = OUTPUT_FIGURES_DIR / "priority_coverage_rate_by_k.png"

RADIUS = 150.0
K_VALUES = [5, 10, 15, 20]
WEIGHT_SCHEMES = ["unweighted", "priority_mild", "priority_strong"]


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


def selected_ids(selected: set[int]) -> str:
    """Format selected candidate IDs for CSV output."""

    return "{" + ", ".join(str(index) for index in sorted(selected)) + "}"


def priority_mask_from_demand_table(demand_points: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask for priority demand points."""

    return demand_points["source_type"].map(is_priority_source_type).to_numpy(dtype=bool)


def run_algorithms(
    items: list[int],
    k: int,
    coverage_sets: dict[int, set[int]],
    n_demand: int,
    demand_weights: np.ndarray,
    weight_scheme: str,
) -> list[tuple[str, dict[str, object]]]:
    """Run Greedy and Lazy Greedy for one weighted coverage scenario."""

    if weight_scheme == "unweighted":
        objective = lambda selected: coverage_objective_geo(coverage_sets, selected, n_demand)
        geo_marginal_gain = lambda x, selected: coverage_marginal_gain_geo(
            coverage_sets,
            x,
            selected,
        )
    else:
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
    marginal_gain = lambda selected, x: geo_marginal_gain(x, selected)

    return [
        ("Greedy", greedy(items, k, objective, marginal_gain=marginal_gain)),
        ("Lazy Greedy", lazy_greedy(items, k, objective)),
    ]


def build_row(
    weight_scheme: str,
    k: int,
    algorithm_name: str,
    result: dict[str, object],
    coverage_sets: dict[int, set[int]],
    distance_matrix: np.ndarray,
    n_demand: int,
    demand_weights: np.ndarray,
    priority_mask: np.ndarray,
) -> dict[str, float | int | str]:
    """Build one scenario result row."""

    selected = set(int(index) for index in result["selected"])
    covered = covered_demand_indices(coverage_sets, selected)
    coverage_count = coverage_objective_geo(coverage_sets, selected, n_demand)
    total_weight = float(np.sum(demand_weights))
    weighted_value = weighted_coverage_objective_geo(
        coverage_sets,
        selected,
        demand_weights,
    )
    priority_total = int(np.sum(priority_mask))
    priority_covered = int(np.sum(priority_mask[list(covered)])) if covered else 0

    return {
        "radius_m": int(RADIUS),
        "k": k,
        "weight_scheme": weight_scheme,
        "algorithm": algorithm_name,
        "objective_value": float(result["value"]),
        "coverage_count": int(coverage_count),
        "coverage_rate": coverage_count / n_demand if n_demand > 0 else 0.0,
        "total_weight": total_weight,
        "weighted_coverage_value": weighted_value,
        "weighted_coverage_rate": weighted_value / total_weight if total_weight > 0 else 0.0,
        "priority_points_total": priority_total,
        "priority_points_covered": priority_covered,
        "priority_coverage_rate": priority_covered / priority_total
        if priority_total > 0
        else 0.0,
        "avg_nearest_distance_m": average_nearest_distance(distance_matrix, selected),
        "max_nearest_distance_m": max_nearest_distance(distance_matrix, selected),
        "eval_count": int(result["eval_count"]),
        "runtime_seconds": float(result["runtime"]),
        "selected_ids": selected_ids(selected),
    }


def build_scenario_table(
    demand_points: pd.DataFrame,
    candidate_points: pd.DataFrame,
) -> pd.DataFrame:
    """Run all weighted coverage scenarios and return the result table."""

    distance_matrix = pairwise_distance_matrix(
        xy_from_table(demand_points),
        xy_from_table(candidate_points),
    )
    coverage_sets = build_coverage_sets(distance_matrix, RADIUS)
    priority_mask = priority_mask_from_demand_table(demand_points)
    items = list(range(len(candidate_points)))
    rows: list[dict[str, float | int | str]] = []

    for weight_scheme in WEIGHT_SCHEMES:
        weighted_demand = assign_demand_weights(demand_points, weight_scheme)
        demand_weights = weighted_demand["weight"].to_numpy(dtype=np.float64)
        for k in K_VALUES:
            if k > len(candidate_points):
                continue
            for algorithm_name, result in run_algorithms(
                items,
                k,
                coverage_sets,
                len(demand_points),
                demand_weights,
                weight_scheme,
            ):
                rows.append(
                    build_row(
                        weight_scheme,
                        k,
                        algorithm_name,
                        result,
                        coverage_sets,
                        distance_matrix,
                        len(demand_points),
                        demand_weights,
                        priority_mask,
                    )
                )

    return pd.DataFrame(rows)


def validate_rates(table: pd.DataFrame) -> list[str]:
    """Validate rate columns from their numerators and denominators."""

    warnings: list[str] = []
    weighted_rates = table["weighted_coverage_value"] / table["total_weight"]
    if not np.allclose(table["weighted_coverage_rate"], weighted_rates):
        warnings.append("weighted_coverage_rate validation failed.")

    priority_rates = table["priority_points_covered"] / table["priority_points_total"]
    if not np.allclose(table["priority_coverage_rate"], priority_rates):
        warnings.append("priority_coverage_rate validation failed.")

    return warnings


def validate_lazy_matches_greedy(table: pd.DataFrame) -> list[str]:
    """Validate Lazy Greedy objective parity against Greedy."""

    warnings: list[str] = []
    for weight_scheme in WEIGHT_SCHEMES:
        for k in K_VALUES:
            subset = table[(table["weight_scheme"] == weight_scheme) & (table["k"] == k)]
            greedy_row = subset[subset["algorithm"] == "Greedy"]
            lazy_row = subset[subset["algorithm"] == "Lazy Greedy"]
            if greedy_row.empty or lazy_row.empty:
                continue
            greedy_value = float(greedy_row.iloc[0]["weighted_coverage_value"])
            lazy_value = float(lazy_row.iloc[0]["weighted_coverage_value"])
            if not np.isclose(greedy_value, lazy_value):
                warnings.append(
                    "Lazy Greedy weighted_coverage_value differs from Greedy "
                    f"for k={k}, weight_scheme={weight_scheme}: "
                    f"{lazy_value} != {greedy_value}."
                )
    return warnings


def selected_set_from_ids(value: str) -> set[int]:
    """Parse the selected_ids display string back to a set."""

    stripped = value.strip("{} ")
    if not stripped:
        return set()
    return {int(part.strip()) for part in stripped.split(",")}


def validate_priority_strong_dominates(
    table: pd.DataFrame,
    coverage_sets: dict[int, set[int]],
    strong_weights: np.ndarray,
) -> list[str]:
    """Check priority_strong selected sets against unweighted under strong weights."""

    warnings: list[str] = []
    greedy_rows = table[table["algorithm"] == "Greedy"]
    for k in K_VALUES:
        unweighted_row = greedy_rows[
            (greedy_rows["k"] == k) & (greedy_rows["weight_scheme"] == "unweighted")
        ]
        strong_row = greedy_rows[
            (greedy_rows["k"] == k) & (greedy_rows["weight_scheme"] == "priority_strong")
        ]
        if unweighted_row.empty or strong_row.empty:
            continue

        unweighted_selected = selected_set_from_ids(str(unweighted_row.iloc[0]["selected_ids"]))
        strong_selected = selected_set_from_ids(str(strong_row.iloc[0]["selected_ids"]))
        unweighted_strong_value = weighted_coverage_objective_geo(
            coverage_sets,
            unweighted_selected,
            strong_weights,
        )
        strong_value = weighted_coverage_objective_geo(
            coverage_sets,
            strong_selected,
            strong_weights,
        )
        if strong_value + 1e-9 < unweighted_strong_value:
            warnings.append(
                "priority_strong selected set has lower priority_strong value than "
                f"unweighted for k={k}: {strong_value} < {unweighted_strong_value}."
            )
    return warnings


def build_report_table(table: pd.DataFrame) -> pd.DataFrame:
    """Build the Greedy weighted-vs-unweighted report table."""

    greedy = table[table["algorithm"] == "Greedy"].copy()
    unweighted_by_k = {
        int(row.k): selected_set_from_ids(str(row.selected_ids))
        for row in greedy[greedy["weight_scheme"] == "unweighted"].itertuples(index=False)
    }

    rows: list[dict[str, float | int | str]] = []
    for row in greedy.itertuples(index=False):
        selected = selected_set_from_ids(str(row.selected_ids))
        unweighted_selected = unweighted_by_k.get(int(row.k), set())
        overlap = len(selected & unweighted_selected)
        overlap_rate = round(overlap / int(row.k), 4) if int(row.k) > 0 else 0.0
        rows.append(
            {
                "k": int(row.k),
                "weight_scheme": str(row.weight_scheme),
                "coverage_count": int(row.coverage_count),
                "coverage_rate": float(row.coverage_rate),
                "weighted_coverage_value": float(row.weighted_coverage_value),
                "weighted_coverage_rate": float(row.weighted_coverage_rate),
                "priority_points_covered": int(row.priority_points_covered),
                "priority_coverage_rate": float(row.priority_coverage_rate),
                "selected_overlap_count": overlap,
                "selected_overlap_rate": overlap_rate,
            }
        )

    return pd.DataFrame(rows)


def save_tables(table: pd.DataFrame, report: pd.DataFrame) -> None:
    """Save scenario and report tables."""

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False)
    LATEX_PATH.write_text(dataframe_to_latex(table), encoding="utf-8")
    report.to_csv(REPORT_CSV_PATH, index=False)
    REPORT_LATEX_PATH.write_text(dataframe_to_latex(report), encoding="utf-8")


def save_metric_plot(table: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> None:
    """Save a weighted scenario metric plot by k."""

    OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for weight_scheme in WEIGHT_SCHEMES:
        subset = table[
            (table["weight_scheme"] == weight_scheme) & (table["algorithm"] == "Greedy")
        ].sort_values("k")
        ax.plot(
            subset["k"],
            subset[metric],
            marker="o",
            linewidth=2,
            label=f"{weight_scheme} - Greedy",
        )
        lazy_subset = table[
            (table["weight_scheme"] == weight_scheme)
            & (table["algorithm"] == "Lazy Greedy")
        ].sort_values("k")
        ax.plot(
            lazy_subset["k"],
            lazy_subset[metric],
            marker="x",
            linestyle="--",
            linewidth=1.5,
            label=f"{weight_scheme} - Lazy",
        )

    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_figures(table: pd.DataFrame) -> None:
    """Save weighted scenario figures."""

    save_metric_plot(
        table,
        "weighted_coverage_value",
        "Weighted coverage value",
        WEIGHTED_VALUE_FIGURE_PATH,
    )
    save_metric_plot(
        table,
        "priority_coverage_rate",
        "Priority coverage rate",
        PRIORITY_RATE_FIGURE_PATH,
    )


def main() -> int:
    """Run weighted coverage scenarios."""

    try:
        demand_points, candidate_points = load_processed_points(DATA_DIR)
        distance_matrix = pairwise_distance_matrix(
            xy_from_table(demand_points),
            xy_from_table(candidate_points),
        )
        coverage_sets = build_coverage_sets(distance_matrix, RADIUS)
        strong_weights = assign_demand_weights(
            demand_points,
            "priority_strong",
        )["weight"].to_numpy(dtype=np.float64)
    except (FileNotFoundError, ValueError) as exc:
        print(
            "Processed OSM data not found or invalid. Run "
            "`python experiments/run_osm_data_collection.py` first. "
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 1

    table = build_scenario_table(demand_points, candidate_points)
    report = build_report_table(table)

    warnings = []
    warnings.extend(validate_rates(table))
    warnings.extend(validate_lazy_matches_greedy(table))
    warnings.extend(validate_priority_strong_dominates(table, coverage_sets, strong_weights))
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    save_tables(table, report)
    save_figures(table)

    print("Weighted coverage scenarios")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")
    print(f"Report CSV saved to: {REPORT_CSV_PATH}")
    print(f"Report LaTeX saved to: {REPORT_LATEX_PATH}")
    print(f"Weighted value figure saved to: {WEIGHTED_VALUE_FIGURE_PATH}")
    print(f"Priority rate figure saved to: {PRIORITY_RATE_FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
