# Submodular Simulation

Lightweight Python project skeleton for the thesis experiments described in
`SPEC.md`.

## Scope

The project is organized for two submodular optimization problems:

- Maximum Coverage
- Facility Location

The codebase is intentionally minimal at this stage. It only provides the
directory structure, module skeletons, and experiment entry points needed to
start implementation.

## Structure

```text
src/
  algorithms.py
  max_coverage.py
  facility_location.py
  plotting.py
experiments/
  run_max_coverage_small.py
  run_facility_location_small.py
  run_runtime_comparison.py
outputs/
  figures/
  tables/
```

## Next Implementation Targets

- Implement brute force, greedy, lazy greedy, and stochastic greedy.
- Add random baselines where required by the SPEC.
- Export CSV and LaTeX tables.
- Save runtime plots and facility location scatter plots.
