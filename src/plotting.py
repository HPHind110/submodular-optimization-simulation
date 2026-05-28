"""Plotting skeletons for experiment outputs.

This module will later generate:

- scatter plots for facility location representatives
- runtime comparison figures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_runtime_plot(results: Any, output_path: str | Path) -> Path:
    """Placeholder function for saving a runtime comparison figure."""

    _ = results
    return Path(output_path)


def save_facility_scatter_plot(data: Any, selected: Any, output_path: str | Path) -> Path:
    """Placeholder function for saving a facility location scatter plot."""

    _ = data
    _ = selected
    return Path(output_path)
