# Bài tập 2 — Đặc trưng góc & Phân vùng ảnh

Project Python thống nhất, tổ chức theo hướng **mỗi thuật toán có một module `.py` riêng và một notebook `.ipynb` để giải thích, kiểm chứng, thực nghiệm**.

## 1. Cây thư mục

```text
Computer_Vision_BT2/
├── README.md
├── TEAM_5_MEMBERS.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── references/
│   ├── De_bai_Bai_tap_2_Thi_giac_may_tinh.pdf
│   ├── Phan_cong_nhiem_vu_nhom.docx
│   └── Bao_cao_thanh_vien_SRG.docx
├── src/
│   └── cv_bt2/
│       ├── __init__.py
│       ├── corner/
│       │   ├── __init__.py
│       │   ├── harris_classic.py       # Harris cổ điển
│       │   ├── harris_laplace.py       # A1: Harris–Laplace
│       │   ├── anms.py                 # A2: ANMS và Top-N
│       │   └── metrics.py              # repeatability
│       ├── segmentation/
│       │   ├── __init__.py
│       │   ├── region_growing.py       # Region Growing cổ điển
│       │   ├── srg.py                  # Seeded Region Growing
│       │   └── metrics.py              # phép đo/kiểm chứng phân vùng
│       ├── data/
│       │   ├── __init__.py
│       │   └── synthetic.py            # dữ liệu tổng hợp tái lập
│       ├── visualization.py
│       └── experiments.py
├── notebooks/
│   ├── 00_project_guide.ipynb
│   ├── 01_harris_classic.ipynb
│   ├── 02_harris_laplace.ipynb
│   ├── 03_anms.ipynb
│   ├── 04_region_growing.ipynb
│   ├── 05_seeded_region_growing.ipynb
│   └── 06_integrated_comparison.ipynb
├── scripts/
│   └── run_all.py
├── tests/
│   └── test_algorithms.py
├── data/
│   └── images/README.md
└── outputs/
    ├── figures/
    └── tables/
```

## 2. Ánh xạ thuật toán ↔ file Python ↔ notebook

| Nội dung | File định nghĩa | Notebook hướng dẫn/test |
|---|---|---|
| Harris cổ điển | `src/cv_bt2/corner/harris_classic.py` | `01_harris_classic.ipynb` |
| Harris–Laplace (A1) | `src/cv_bt2/corner/harris_laplace.py` | `02_harris_laplace.ipynb` |
| ANMS (A2) | `src/cv_bt2/corner/anms.py` | `03_anms.ipynb` |
| Region Growing cổ điển | `src/cv_bt2/segmentation/region_growing.py` | `04_region_growing.ipynb` |
| Seeded Region Growing | `src/cv_bt2/segmentation/srg.py` | `05_seeded_region_growing.ipynb` |

Notebook `06_integrated_comparison.ipynb` tổng hợp ba cặp so sánh chính theo đề. Notebook `00_project_guide.ipynb` hướng dẫn cấu trúc và cách chạy.

## 3. Cài đặt môi trường

Khuyến nghị Python 3.10 trở lên. Mở terminal tại thư mục gốc project. Phần cài đặt thuật toán tự triển khai bằng NumPy/Python; SciPy chỉ được dùng cho `gaussian_filter` và đạo hàm Gaussian. NMS, repeatability, nội suy ảnh, phép xoay và ánh xạ nhãn được tự cài đặt. OpenCV không bắt buộc; notebook Harris có cell đối chiếu tùy chọn với `cv2.cornerHarris` nếu máy đã cài `opencv-python`.

```bash
# Tạo môi trường ảo
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate

# Cài thư viện và project ở chế độ editable
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. Thứ tự chạy đề xuất

1. `notebooks/01_harris_classic.ipynb`
2. `notebooks/02_harris_laplace.ipynb`
3. `notebooks/03_anms.ipynb`
4. `notebooks/04_region_growing.ipynb`
5. `notebooks/05_seeded_region_growing.ipynb`
6. `notebooks/06_integrated_comparison.ipynb`

Mở Jupyter từ **thư mục gốc project**:

```bash
jupyter lab
```

Notebook có đoạn xác định `ROOT` để import package khi mở từ thư mục gốc hoặc từ `notebooks/`.

## 5. Chạy kiểm thử và tạo kết quả bằng terminal

```bash
pytest -q
python scripts/run_all.py
```

`pytest` kiểm tra các hợp đồng chính: Harris tìm góc bàn cờ; Harris–Laplace trả đúng định dạng; ANMS và Top-N trả cùng N điểm; Region Growing và SRG tạo nhãn đúng; SRG full partition, số vùng, frame, Lazy/Frozen, tie-breaking và seed sensitivity.

`run_all.py` sinh ảnh PNG trong `outputs/figures/` và bảng CSV trong `outputs/tables/`.

## 6. Tóm tắt cách dùng các hàm

### Harris cổ điển

```python
from cv_bt2.corner import harris_classic
points = harris_classic(img, sigma_i=2.0, sigma_d=1.4, k=0.04, thresh_rel=0.01)
# points: (N,3) -> row, col, response
```

### Harris–Laplace

```python
from cv_bt2.corner import scale_pyramid, harris_laplace
sigmas = scale_pyramid(sigma0=1.2, xi=1.2, n_levels=10)
points = harris_laplace(img, sigmas, log_thresh=0.0)
# points: (N,4) -> row, col, sigma, response
```

### ANMS

```python
from cv_bt2.corner import anms, top_n, grid_occupancy
selected = anms(harris_points, n_keep=100, c_robust=0.9)
coverage = grid_occupancy(selected, img.shape, grid_size=8)
```

### Region Growing

```python
from cv_bt2.segmentation import region_growing_classic
labels = region_growing_classic(img, seeds, T=25, order="stack")
# labels: int32 (H,W), -1 là pixel chưa được gán
```

### SRG

```python
from cv_bt2.segmentation import srg, srg_step_frames, seed_sensitivity
labels, boundary_mask = srg(
    img, seeds, connectivity=4, stale="lazy", tie="insertion"
)
frames = srg_step_frames(img, seeds, n_frames=8)
```

## 7. Các điểm kỹ thuật cần hiểu khi thuyết trình

- **Harris:** `sigma_d` làm trơn trước gradient; `sigma_i` tích phân các tích gradient. Harris response dùng `det(M)-k*trace(M)^2`.
- **Harris–Laplace:** không lấy argmax Harris theo scale; giữ cực đại địa phương của LoG chuẩn hóa theo trục scale.
- **ANMS:** không phát hiện góc mới; chỉ chọn lại tập Harris để cân bằng độ phủ không gian. Luôn so sánh Top-N và ANMS với cùng N.
- **Region Growing:** `T` quyết định mức tương đồng; Stack và Queue khác thứ tự duyệt, và vùng có thể còn pixel `-1`.
- **SRG:** min-heap kết nạp ứng viên delta nhỏ nhất. Lazy cập nhật theo mean hiện tại; Frozen dùng seed ban đầu. Tie-breaking có thể tác động kết quả, nhất là ảnh phẳng.

## 8. Lưu ý diễn giải kết quả

1. Ảnh mặc định là **ảnh tổng hợp do project tự tạo**, có seed ngẫu nhiên cố định để chạy tái lập. Thay ảnh thật có thể làm số liệu thay đổi.
2. Không chép trước số liệu vào báo cáo. Hãy chạy notebook, lưu output và trích đúng số liệu/biểu đồ được sinh ở máy của nhóm.
3. Với Region Growing, khi so sánh với SRG hãy báo cáo riêng pixel chưa gán của RG. Phép so sánh phân vùng trong `segmentation.metrics` bỏ qua khác biệt ID nhãn và tính trên pixel được cả hai kết quả gán nhãn.
4. `repeatability` ở đây tính mỗi điểm A được tìm lại nếu có ít nhất một điểm B trong bán kính `eps`; một điểm B có thể ghép với nhiều điểm A, đúng cách đo tọa độ lân cận cơ bản mô tả trong tài liệu.
5. SRG dùng độ tương phản không âm `||I(x)-mean(A_i)||`; với ảnh xám đây là trị tuyệt đối của hiệu. Ảnh màu sử dụng chuẩn L2.
6. `boundary_mask` đánh dấu cả hai phía pixel của mọi cặp lân cận có nhãn khác nhau.
7. Seed-order invariance là phép kiểm chứng thực nghiệm theo ảnh và tie policy đang dùng, không phải bảo đảm tuyệt đối cho mọi dữ liệu.

## 9. Phân công làm việc nhóm 5 người

Project được tổ chức theo đúng tài liệu phân công 5 thành viên: (1) Harris cổ điển, (2) Harris–Laplace, (3) ANMS, (4) Region Growing cổ điển, (5) Seeded Region Growing. Bảng phân công chi tiết, mục tiêu, thí nghiệm và checklist bàn giao nằm trong `TEAM_5_MEMBERS.md`. Mỗi thành viên có module `.py` và notebook `.ipynb` riêng.

## 10. Những gì được kế thừa và những gì đã bổ sung

- Harris cổ điển được tổ chức lại từ `harris_functions.py` đã gửi; bổ sung xử lý plateau NMS tất định và kiểm tra đầu vào.
- Phần Harris–Laplace được tách thành module Python để notebook không định nghĩa lại toàn bộ hàm. Cấu trúc các thử nghiệm A1 trong notebook thành viên 2 được dùng làm căn cứ, nhưng ảnh `anh/facade_1x.png` và `anh/facade_3x.png` không có trong file gửi nên project thay bằng ảnh bàn cờ tổng hợp tái lập.
- Region Growing được tổ chức lại từ `region_growing.py` đã gửi.
- Phần SRG được tái xây dựng vì `srg.py` mà báo cáo thành viên 5 nhắc tới không nằm trong các file đính kèm. Các yêu cầu Lazy/Frozen, tie, connectivity, frames, seed sensitivity được triển khai trong `segmentation/srg.py`.
- Module metrics, ảnh tổng hợp, notebooks, tests và script chạy toàn bộ thí nghiệm được bổ sung để gom project thành một cấu trúc thống nhất.

## 11. Tài liệu gốc nhóm đã cung cấp

Cấu trúc và các chức năng được đối chiếu với `Computer_Vision 45.pdf`, tài liệu phân công 5 thành viên, notebook Harris–Laplace, `harris_functions.py`, `region_growing.py`, các script thí nghiệm và báo cáo thành viên 5. Bản PDF đề bài, phân công và báo cáo SRG gốc được đặt trong `references/` để nhóm tra cứu. Hãy dùng đề bài PDF làm chuẩn cuối cùng nếu giảng viên có hướng dẫn cập nhật.
"# thigiac" 
