"""Run the small Maximum Coverage experiment from ``SPEC.md``.

The script evaluates:

- brute force
- greedy
- random baseline

Results are exported to CSV and LaTeX under ``outputs/tables``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import brute_force, greedy, random_baseline
from src.max_coverage import (
    coverage_marginal_gain,
    coverage_objective,
    get_small_coverage_instance,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
CSV_PATH = OUTPUT_DIR / "max_coverage_small.csv"
LATEX_PATH = OUTPUT_DIR / "max_coverage_small.tex"


def format_selected(selected: set[str]) -> str:
    """Return a stable display string for a selected set."""

    return "{" + ", ".join(sorted(selected)) + "}"


def build_results_table() -> pd.DataFrame:
    """Run the small Maximum Coverage algorithms and return a summary table."""

    sets, items, k = get_small_coverage_instance()

    objective = lambda selected: coverage_objective(sets, selected)
    marginal_gain = lambda selected, x: coverage_marginal_gain(sets, x, selected)

    brute_force_result = brute_force(items, k, objective)
    greedy_result = greedy(items, k, objective, marginal_gain=marginal_gain)
    random_result = random_baseline(items, k, objective, n_trials=1000, seed=42)

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

    return pd.DataFrame(rows)


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


def save_outputs(table: pd.DataFrame) -> None:
    """Write CSV and LaTeX outputs to ``outputs/tables``."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_PATH, index=False)
    latex_table = dataframe_to_latex(table)
    LATEX_PATH.write_text(latex_table, encoding="utf-8")


def print_summary(table: pd.DataFrame) -> None:
    """Print a concise experiment summary to the terminal."""

    print("Maximum Coverage small experiment")
    print(table.to_string(index=False))
    print(f"CSV saved to: {CSV_PATH}")
    print(f"LaTeX saved to: {LATEX_PATH}")


def main() -> None:
    """Run the experiment and export results."""

    table = build_results_table()
    save_outputs(table)
    print_summary(table)


if __name__ == "__main__":
    main()
