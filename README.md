# Submodular Optimization for Public Wi-Fi Coverage

Du an nay mo phong bai toan phu song Wi-Fi cong cong tai khu vuc Hoan Kiem,
Ha Noi bang du lieu OpenStreetMap. Mo hinh chinh la Maximum Coverage tren do
thi/khong gian dia ly, voi hai tap vi tri ung vien va hai bien the objective:

1. Unweighted Maximum Coverage.
2. Weighted Maximum Coverage theo muc do uu tien cua demand points.

Project phuc vu truc tiep cho Chuong 4 cua do an. Cac toy synthetic
experiments va Facility Location cu da duoc chuyen vao `archive/`.

## 1. Setup

Tao va kich hoat moi truong ao:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cai dependencies:

```powershell
pip install -r requirements.txt
```

## 2. Pipeline Chinh

Lay va xu ly du lieu OSM:

```powershell
python experiments/run_osm_data_collection.py --place "Hoan Kiem, Hanoi, Vietnam" --max-demand 1000 --max-candidates 400 --include-road-nodes --max-road-node-candidates 400 --seed 42
```

Chay so sanh hai tap candidate voi cac ban kinh phu song:

```powershell
python experiments/run_candidate_scenarios.py --radius 100
python experiments/run_candidate_scenarios.py --radius 150
python experiments/run_candidate_scenarios.py --radius 200
```

Validate candidate scenarios:

```powershell
python experiments/validate_candidate_scenarios.py --radii 100 150 200
```

Chay weighted coverage scenarios:

```powershell
python experiments/run_weighted_coverage_scenarios.py
```

Validate weighted coverage:

```powershell
python experiments/validate_weighted_coverage.py
```

Tao bang va hinh rut gon cho bao cao:

```powershell
python experiments/build_report_tables.py
python experiments/build_report_figures.py
```

## 3. Outputs Chinh

Du lieu da xu ly:

```text
data/processed/demand_points.csv
data/processed/candidate_points.csv
data/processed/candidate_points_bus_stop_only.csv
data/processed/candidate_points_road_nodes.csv
```

Bang report-level:

```text
outputs/tables/candidate_scenario_comparison_R100.csv
outputs/tables/candidate_scenario_comparison_R150.csv
outputs/tables/candidate_scenario_comparison_R200.csv
outputs/tables/weighted_coverage_scenarios.csv
outputs/tables/report_weighted_vs_unweighted_R150.csv
outputs/tables/report_candidate_coverage_by_radius.csv
outputs/tables/report_lazy_efficiency_R150.csv
```

Hinh report-level:

```text
outputs/figures/candidate_scenario_coverage_R100.png
outputs/figures/candidate_scenario_coverage_R150.png
outputs/figures/candidate_scenario_coverage_R200.png
outputs/figures/weighted_coverage_value_by_k.png
outputs/figures/priority_coverage_rate_by_k.png
outputs/figures/report_coverage_rate_by_k_all_radii.png
outputs/figures/report_lazy_eval_reduction_R150.png
outputs/figures/report_candidate_scenario_map_R150_k10.png
```

## 4. Archive

Thu muc `archive/` chua cac experiment cu khong dung trong report chinh:

- `archive/synthetic_experiments/`: toy Maximum Coverage, synthetic Gaussian
  Facility Location, runtime synthetic.
- `archive/old_experiments/`: cac script real OSM thu nghiem cu co output bi
  ghi de hoac khong con nam trong pipeline chinh.
- `archive/old_outputs/`: bang va hinh cu khong dung truc tiep trong Chuong 4.

Khong file nao bi xoa vinh vien trong buoc don repo.
