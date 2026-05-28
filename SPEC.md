\# SPEC: Mô phỏng tối ưu dưới mô-đun cho đồ án



\## Mục tiêu



Xây dựng mã Python để mô phỏng các thuật toán cho hai bài toán:



1\. Maximum Coverage, dùng cho bài toán phủ sóng.

2\. Facility Location, dùng cho bài toán lựa chọn tập đại diện.



Mục tiêu của mô phỏng là phục vụ Chương 4 của đồ án toán ứng dụng, không phải xây dựng hệ thống AI phức tạp.



\## Yêu cầu chung



\- Code rõ ràng, dễ đọc.

\- Không dùng framework nặng.

\- Dùng numpy, pandas, matplotlib, scikit-learn nếu cần.

\- Mỗi script chạy độc lập.

\- Kết quả được lưu vào thư mục outputs/tables và outputs/figures.

\- Có đo thời gian chạy và số lần đánh giá hàm mục tiêu hoặc lợi ích biên.



\## Thuật toán cần cài đặt



\### 1. Brute force



Dùng cho dữ liệu nhỏ để tìm nghiệm tối ưu.



Input:

\- items

\- k

\- objective function



Output:

\- selected set

\- objective value

\- number of evaluations

\- runtime



\### 2. Greedy



Ở mỗi bước chọn phần tử có lợi ích biên lớn nhất.



Input:

\- items

\- k

\- marginal gain function



Output:

\- selected set

\- objective value

\- number of evaluations

\- runtime



\### 3. Lazy Greedy



Dùng priority queue để giảm số lần đánh giá lại lợi ích biên.



Yêu cầu:

\- Cho cùng nghiệm với greedy cơ bản nếu không có khác biệt tie-breaking.

\- Ghi lại số lần đánh giá objective.



\### 4. Stochastic Greedy



Ở mỗi bước lấy mẫu một tập con ứng viên rồi chọn phần tử tốt nhất trong mẫu.



Input:

\- epsilon

\- seed



Kích thước mẫu:

r = (n / k) \* ln(1 / epsilon)



\## Bài toán Maximum Coverage



Dữ liệu ví dụ:

U = {1,2,3,4,5,6,7}



A1 = {1,2,3}

A2 = {2,3,4}

A3 = {4,5}

A4 = {5,6,7}

A5 = {1,7}



k = 2



Cần chạy:

\- brute force

\- greedy

\- random baseline



Xuất bảng CSV và LaTeX.



\## Bài toán Facility Location



Sinh dữ liệu 2D gồm 2 hoặc 3 cụm Gaussian.



Similarity:

w\_ij = exp(-||x\_i - x\_j||^2 / (2 sigma^2))



Hàm mục tiêu:

f(S) = sum\_i max\_{j in S} w\_ij



Cần chạy:

\- Với dữ liệu nhỏ: brute force, greedy, random baseline.

\- Với dữ liệu lớn: greedy, lazy greedy, stochastic greedy.



Xuất:

\- bảng CSV

\- bảng LaTeX

\- hình scatter plot các cụm dữ liệu và điểm đại diện được chọn

\- biểu đồ runtime theo số điểm dữ liệu



\## Không làm



\- Không triển khai Continual Learning.

\- Không train model AI.

\- Không dùng deep learning.

\- Không dùng dữ liệu ngoài.

