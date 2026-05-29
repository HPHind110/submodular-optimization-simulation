"""Build compact Chapter 4 report tables from validated experiment CSVs."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
RADII = [100, 150, 200]


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
        values = [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def load_candidate_tables() -> pd.DataFrame:
    """Load candidate scenario tables for all report radii."""

    frames = []
    for radius in RADII:
        path = OUTPUT_TABLES_DIR / f"candidate_scenario_comparison_R{radius}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing candidate scenario table: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def save_table(table: pd.DataFrame, name: str) -> tuple[Path, Path]:
    """Save a CSV and LaTeX report table."""

    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_TABLES_DIR / f"{name}.csv"
    latex_path = OUTPUT_TABLES_DIR / f"{name}.tex"
    table.to_csv(csv_path, index=False)
    latex_path.write_text(dataframe_to_latex(table), encoding="utf-8")
    return csv_path, latex_path


def build_candidate_coverage_by_radius(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Build a report table comparing scenario coverage across radii."""

    greedy = candidate_table[candidate_table["algorithm"] == "Greedy"].copy()
    return greedy[
        [
            "radius_m",
            "k",
            "scenario",
            "n_candidate",
            "coverage_count",
            "coverage_rate",
            "avg_nearest_distance_m",
            "max_nearest_distance_m",
        ]
    ].sort_values(["radius_m", "k", "scenario"])


def build_lazy_efficiency(candidate_table: pd.DataFrame) -> pd.DataFrame:
    """Build a report table for Lazy Greedy evaluation savings at R=150."""

    radius_table = candidate_table[candidate_table["radius_m"] == 150]
    rows: list[dict[str, float | int | str]] = []
    for (scenario, k), subset in radius_table.groupby(["scenario", "k"], sort=False):
        greedy = subset[subset["algorithm"] == "Greedy"].iloc[0]
        lazy = subset[subset["algorithm"] == "Lazy Greedy"].iloc[0]
        greedy_evals = int(greedy["eval_count"])
        lazy_evals = int(lazy["eval_count"])
        rows.append(
            {
                "scenario": scenario,
                "k": int(k),
                "coverage_count": int(greedy["coverage_count"]),
                "greedy_eval_count": greedy_evals,
                "lazy_eval_count": lazy_evals,
                "eval_reduction_rate": 1.0 - (lazy_evals / greedy_evals),
                "greedy_runtime_seconds": float(greedy["runtime_seconds"]),
                "lazy_runtime_seconds": float(lazy["runtime_seconds"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["scenario", "k"])


def main() -> int:
    """Build all report tables."""

    try:
        candidate_table = load_candidate_tables()
        coverage_table = build_candidate_coverage_by_radius(candidate_table)
        lazy_table = build_lazy_efficiency(candidate_table)
        coverage_paths = save_table(coverage_table, "report_candidate_coverage_by_radius")
        lazy_paths = save_table(lazy_table, "report_lazy_efficiency_R150")
    except Exception as exc:
        print(f"Could not build report tables. Error: {exc}", file=sys.stderr)
        return 1

    print("Report tables built")
    print(f"Candidate coverage table: {coverage_paths[0]}")
    print(f"Candidate coverage LaTeX: {coverage_paths[1]}")
    print(f"Lazy efficiency table: {lazy_paths[0]}")
    print(f"Lazy efficiency LaTeX: {lazy_paths[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
