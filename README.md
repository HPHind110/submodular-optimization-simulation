````markdown
# Submodular Optimization Simulation

Dự án này cài đặt và mô phỏng một số thuật toán cho các bài toán tối ưu dưới mô-đun trong đồ án **"Một số vấn đề tối ưu trên đồ thị và ứng dụng"**.

Trọng tâm của mô phỏng gồm:

1. Bài toán bao phủ tối đa (Maximum Coverage), dùng để mô hình hóa bài toán phủ sóng.
2. Bài toán định vị cơ sở (Facility Location), dùng để mô hình hóa bài toán lựa chọn tập đại diện.
3. So sánh một số biến thể của thuật toán tham lam, gồm Greedy, Lazy Greedy và Stochastic Greedy.

## 1. Cấu trúc thư mục

```text
submodular_simulation/
│
├── src/
│   ├── algorithms.py
│   ├── max_coverage.py
│   ├── facility_location.py
│   └── plotting.py
│
├── experiments/
│   ├── run_max_coverage_small.py
│   ├── run_facility_location_small.py
│   └── run_runtime_comparison.py
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── SPEC.md
├── requirements.txt
├── README.md
└── .gitignore
````

## 2. Cài đặt môi trường

Tạo môi trường ảo Python:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

Nếu chưa có file `requirements.txt`, có thể cài trực tiếp:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 3. Chạy thí nghiệm

Chạy thí nghiệm cho bài toán bao phủ tối đa:

```bash
python experiments/run_max_coverage_small.py
```

Chạy thí nghiệm cho bài toán định vị cơ sở trên dữ liệu nhỏ:

```bash
python experiments/run_facility_location_small.py
```

Chạy thí nghiệm so sánh thời gian chạy giữa Greedy, Lazy Greedy và Stochastic Greedy:

```bash
python experiments/run_runtime_comparison.py
```

## 4. Kết quả đầu ra

Các kết quả được lưu trong thư mục `outputs/`.

### Bảng kết quả

```text
outputs/tables/
```

Thư mục này chứa các bảng kết quả ở dạng `.csv` và `.tex`.

### Hình minh họa

```text
outputs/figures/
```

Thư mục này chứa các hình minh họa và biểu đồ thực nghiệm.

## 5. Nội dung mô phỏng

### Maximum Coverage

Bài toán bao phủ tối đa được dùng để mô hình hóa bài toán phủ sóng. Mục tiêu là chọn tối đa (k) tập con sao cho số phần tử được bao phủ là lớn nhất.

Các thuật toán được sử dụng:

* Brute Force
* Greedy
* Random Baseline

### Facility Location

Bài toán định vị cơ sở được dùng để mô hình hóa bài toán lựa chọn tập đại diện. Hàm mục tiêu có dạng:

[
f(S)=\sum_i \max_{j\in S} w_{ij}.
]

Các thuật toán được sử dụng:

* Brute Force
* Greedy
* Random Baseline
* Lazy Greedy
* Stochastic Greedy

## 6. Ghi chú

Mô phỏng này tập trung vào khía cạnh toán ứng dụng và tối ưu tổ hợp. Dự án không triển khai hệ thống học máy hoặc học liên tục hoàn chỉnh.

```
```
