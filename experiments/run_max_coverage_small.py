"""Run the small Maximum Coverage experiment from the SPEC.

This script is intended to:

- build the toy Maximum Coverage instance
- run brute force, greedy, and random baseline
- export CSV and LaTeX tables to outputs/tables
"""

from __future__ import annotations

from src.max_coverage import build_small_instance


def main() -> None:
    """Entry point for the small Maximum Coverage experiment."""

    instance = build_small_instance()
    print("Maximum Coverage small experiment skeleton")
    print(f"Universe size: {len(instance['universe'])}, k={instance['k']}")


if __name__ == "__main__":
    main()
