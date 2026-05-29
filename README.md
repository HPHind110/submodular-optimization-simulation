# Submodular Optimization for Public Wi-Fi Coverage

Dự án mô phỏng bài toán lựa chọn vị trí đặt trạm Wi-Fi công cộng tại khu vực Hoàn Kiếm, Hà Nội dựa trên dữ liệu OpenStreetMap. Bài toán chính được mô hình hóa dưới dạng Maximum Coverage trên dữ liệu địa lý: chọn tối đa `k` vị trí ứng viên sao cho số điểm nhu cầu được phủ trong bán kính cho trước là lớn nhất.

Dự án phục vụ phần thực nghiệm của Chương 4 trong đồ án, tập trung vào ba câu hỏi:

1. Mở rộng tập vị trí ứng viên có cải thiện khả năng phủ sóng hay không?
2. Lazy Greedy có giảm chi phí tính toán so với Greedy cơ bản hay không?
3. Weighted Coverage có giúp ưu tiên các điểm nhu cầu quan trọng hơn hay không?

Các thí nghiệm toy/synthetic và các phiên bản cũ không còn dùng trong báo cáo chính đã được chuyển vào thư mục `archive/`.

---

## 1. Cài đặt

Tạo môi trường ảo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài các thư viện cần thiết:

```powershell
pip install -r requirements.txt
```

---

## 2. Cấu trúc chính

```text
src/
  algorithms.py                 # Greedy, Lazy Greedy và các thuật toán phụ
  osm_data.py                   # Thu thập và tiền xử lý dữ liệu OpenStreetMap
  geo_metrics.py                # Tính khoảng cách và các metric địa lý
  geo_coverage.py               # Objective và marginal gain cho Maximum Coverage
  plotting.py                   # Các hàm vẽ hình báo cáo

experiments/
  run_osm_data_collection.py          # Thu thập và xử lý dữ liệu OSM
  run_candidate_scenarios.py          # So sánh bus_stop_only và road_nodes
  run_weighted_coverage_scenarios.py  # Chạy thí nghiệm Weighted Coverage
  validate_candidate_scenarios.py     # Kiểm tra tính hợp lệ của candidate scenarios
  validate_weighted_coverage.py       # Kiểm tra tính hợp lệ của weighted scenarios
  build_report_tables.py              # Tạo bảng rút gọn cho báo cáo
  build_report_figures.py             # Tạo hình rút gọn cho báo cáo

data/processed/
  demand_points.csv
  candidate_points.csv
  candidate_points_bus_stop_only.csv
  candidate_points_road_nodes.csv

outputs/
  tables/
  figures/

archive/
  synthetic_experiments/
  old_experiments/
  old_outputs/
```

---

## 3. Pipeline thực nghiệm

### 3.1. Thu thập và tiền xử lý dữ liệu OSM

```powershell
python experiments/run_osm_data_collection.py --place "Hoan Kiem, Hanoi, Vietnam" --max-demand 1000 --max-candidates 400 --include-road-nodes --max-road-node-candidates 400 --seed 42
```

Script này tạo các file dữ liệu đã xử lý:

```text
data/processed/demand_points.csv
data/processed/candidate_points.csv
data/processed/candidate_points_bus_stop_only.csv
data/processed/candidate_points_road_nodes.csv
```

Trong đó:

- `demand_points.csv`: các điểm nhu cầu thu thập từ OSM, ví dụ quán cà phê, nhà hàng, trường học, bệnh viện, thư viện, điểm du lịch, điểm dừng xe buýt.
- `candidate_points_bus_stop_only.csv`: các vị trí ứng viên chỉ gồm điểm dừng giao thông công cộng.
- `candidate_points_road_nodes.csv`: tập ứng viên mở rộng, gồm điểm dừng giao thông công cộng và các nút trên mạng đường.

Các tọa độ được chuyển sang hệ tọa độ projected để tính khoảng cách theo mét.

---

### 3.2. So sánh hai tập vị trí ứng viên

Chạy thí nghiệm với ba bán kính phủ sóng:

```powershell
python experiments/run_candidate_scenarios.py --radius 100
python experiments/run_candidate_scenarios.py --radius 150
python experiments/run_candidate_scenarios.py --radius 200
```

Thí nghiệm so sánh hai kịch bản:

1. `bus_stop_only`: chỉ chọn vị trí từ các điểm dừng giao thông công cộng.
2. `road_nodes`: chọn vị trí từ tập ứng viên mở rộng gồm điểm dừng giao thông công cộng và các nút mạng đường.

Các thuật toán được dùng trong báo cáo chính:

- Greedy
- Lazy Greedy

Kiểm tra kết quả:

```powershell
python experiments/validate_candidate_scenarios.py --radii 100 150 200
```

---

### 3.3. Chạy thí nghiệm Weighted Coverage

```powershell
python experiments/run_weighted_coverage_scenarios.py
```

Thí nghiệm này xét ba kịch bản trọng số:

1. `unweighted`: mọi điểm nhu cầu có trọng số bằng 1.
2. `priority_mild`: ưu tiên nhẹ các địa điểm công cộng như bệnh viện, trường học, thư viện, bảo tàng, điểm du lịch.
3. `priority_strong`: tăng mạnh trọng số cho các điểm nhu cầu quan trọng.

Mục tiêu là kiểm tra xem việc đưa trọng số vào hàm mục tiêu có làm thay đổi nghiệm theo hướng ưu tiên các điểm quan trọng hơn hay không.

Kiểm tra kết quả:

```powershell
python experiments/validate_weighted_coverage.py
```

---

### 3.4. Tạo bảng và hình cho báo cáo

```powershell
python experiments/build_report_tables.py
python experiments/build_report_figures.py
```

---

## 4. Output chính

### 4.1. Bảng kết quả

```text
outputs/tables/candidate_scenario_comparison_R100.csv
outputs/tables/candidate_scenario_comparison_R150.csv
outputs/tables/candidate_scenario_comparison_R200.csv

outputs/tables/report_candidate_coverage_by_radius.csv
outputs/tables/report_lazy_efficiency_R150.csv

outputs/tables/weighted_coverage_scenarios.csv
outputs/tables/report_weighted_vs_unweighted_R150.csv
```

Ý nghĩa chính:

- `report_candidate_coverage_by_radius.csv`: so sánh tỉ lệ phủ giữa `bus_stop_only` và `road_nodes` theo các bán kính và giá trị `k`.
- `report_lazy_efficiency_R150.csv`: so sánh số lần đánh giá hàm mục tiêu và thời gian chạy giữa Greedy và Lazy Greedy tại bán kính 150m.
- `report_weighted_vs_unweighted_R150.csv`: so sánh nghiệm không trọng số và các nghiệm có trọng số ưu tiên.

---

### 4.2. Hình kết quả

```text
outputs/figures/candidate_scenario_coverage_R100.png
outputs/figures/candidate_scenario_coverage_R150.png
outputs/figures/candidate_scenario_coverage_R200.png

outputs/figures/report_coverage_rate_by_k_all_radii.png
outputs/figures/report_lazy_eval_reduction_R150.png
outputs/figures/report_candidate_scenario_map_R150_k10.png

outputs/figures/weighted_coverage_value_by_k.png
outputs/figures/priority_coverage_rate_by_k.png
```

Các hình này được dùng để minh họa:

- ảnh hưởng của bán kính phủ sóng và số trạm được chọn;
- sự khác biệt giữa hai tập vị trí ứng viên;
- mức giảm số lần đánh giá của Lazy Greedy;
- tác động của trọng số ưu tiên trong Weighted Coverage.

---

## 5. Validation

Trước khi sử dụng kết quả trong báo cáo, cần chạy đầy đủ:

```powershell
python experiments/validate_candidate_scenarios.py --radii 100 150 200
python experiments/validate_weighted_coverage.py
```

Kết quả chỉ nên được xem là hợp lệ khi các validation đều pass.

Các điều kiện kiểm tra chính gồm:

- `coverage_rate` khớp với `coverage_count / n_demand`;
- tỉ lệ phủ không giảm khi tăng `k`;
- Lazy Greedy cho cùng giá trị nghiệm với Greedy trong các kịch bản xét;
- Lazy Greedy dùng ít hoặc bằng số lần đánh giá hàm mục tiêu so với Greedy;
- `road_nodes` có số lượng candidate lớn hơn `bus_stop_only`;
- các tỉ lệ trong Weighted Coverage được tính nhất quán.

---

## 6. Ghi chú về phạm vi

Dự án này là mô phỏng thực nghiệm phục vụ đồ án, không phải hệ thống triển khai Wi-Fi thực tế.

Một số giả định chính:

- Dữ liệu được lấy từ OpenStreetMap nên phụ thuộc vào mức độ đầy đủ của OSM tại khu vực được xét.
- Khoảng cách được tính theo tọa độ projected dạng mét.
- Điểm nhu cầu được xem là được phủ nếu nằm trong bán kính phục vụ `R` của ít nhất một vị trí được chọn.
- Các trọng số trong Weighted Coverage là các kịch bản ưu tiên giả định, không phải dữ liệu lưu lượng người dùng thực tế.

---

## 7. Archive

Thư mục `archive/` chứa các file cũ không còn dùng trong pipeline chính:

```text
archive/synthetic_experiments/
archive/old_experiments/
archive/old_outputs/
```

Các file này được giữ lại để tham khảo lịch sử phát triển, nhưng không dùng trực tiếp trong báo cáo chính.
