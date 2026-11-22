# PHÂN CÔNG PROJECT — NHÓM 5 THÀNH VIÊN

Phân công dưới đây bám theo tài liệu `Phan_cong_nhiem_vu_5_thanh_vien_Bai_tap_2.docx`: 3 thành viên làm Phần A và 2 thành viên làm Phần B. Mỗi người phụ trách lý thuyết, code, kiểm chứng, thí nghiệm và giải thích kết quả trong phạm vi được giao; không có thành viên chỉ làm tổng hợp.

## 1. Bảng phân công và file bàn giao

| Thành viên | Phạm vi được giao | Module Python | Notebook phụ trách |
|---|---|---|---|
| **1** | Harris cổ điển và nền tảng Phần A | `src/cv_bt2/corner/harris_classic.py` | `notebooks/01_harris_classic.ipynb` |
| **2** | A1 — Harris–Laplace | `src/cv_bt2/corner/harris_laplace.py` | `notebooks/02_harris_laplace.ipynb` |
| **3** | A2 — Adaptive Non-Maximal Suppression (ANMS) | `src/cv_bt2/corner/anms.py` | `notebooks/03_anms.ipynb` |
| **4** | Region Growing cổ điển | `src/cv_bt2/segmentation/region_growing.py` | `notebooks/04_region_growing.ipynb` |
| **5** | Seeded Region Growing (SRG) | `src/cv_bt2/segmentation/srg.py` | `notebooks/05_seeded_region_growing.ipynb` |

Các module dùng chung:
- Đo repeatability: `src/cv_bt2/corner/metrics.py`.
- Đo phân vùng: `src/cv_bt2/segmentation/metrics.py`.
- Ảnh thử tổng hợp: `src/cv_bt2/data/synthetic.py`.
- Hàm vẽ: `src/cv_bt2/visualization.py`.
- Thí nghiệm tích hợp ba cặp so sánh: `notebooks/06_integrated_comparison.ipynb` và `src/cv_bt2/experiments.py`.

## 2. Nhiệm vụ cụ thể từng người

### Thành viên 1 — Harris cổ điển
- Nắm gradient `Ix, Iy`, structure tensor, hai trị riêng, Harris response `R = det(M) - k·trace(M)^2`.
- Phân biệt `sigma_d` (làm mờ trước đạo hàm) và `sigma_i` (tích phân các tích đạo hàm).
- Kiểm chứng trên bàn cờ; thử quay ảnh 30°; đối chiếu `cv2.cornerHarris` ở phần kiểm chứng nếu có OpenCV.
- Quét `sigma_i`, `sigma_d`, `k`, `thresh_rel`; báo cáo số điểm và hình ảnh.
- Bàn giao đầu ra Harris ổn định để thành viên 2 và 3 dùng chung.

### Thành viên 2 — Harris–Laplace
- Nắm tháp tỷ lệ `sigma_n = xi^n sigma_0`, LoG chuẩn hóa `sigma²|Lxx + Lyy|` và cực đại địa phương theo trục tỷ lệ.
- Dùng các hàm Harris nền tảng từ module thành viên 1.
- So sánh Harris và Harris–Laplace trên cùng hình ở kích thước gốc và phóng to 3×; đo repeatability.
- Quét `xi`, `n_levels`, `log_thresh`; giải thích tỷ lệ được chọn và ứng viên bị loại.
- Bàn giao điểm `(row, col, sigma, response)`, hình minh họa và bảng/đồ thị.

### Thành viên 3 — ANMS
- Nắm bán kính ức chế `r_i`, điều kiện mạnh hơn `f(x_i) < c_robust f(x_j)` và ý nghĩa mặc định `c_robust=0.9`.
- So sánh ANMS với Top-N trên cùng tập điểm Harris và cùng số lượng N.
- Đo `grid_occupancy`; vẽ hai tập điểm trên cùng ảnh.
- Quét `c_robust` từ 0.5 đến 1.0 và quét N; giải thích vì sao điểm yếu nhưng cô lập vẫn có thể được giữ.
- Bàn giao hai hàm `anms`, `top_n`, bảng độ phủ và biểu đồ.

### Thành viên 4 — Region Growing cổ điển
- Nắm tiêu chí kết nạp `|I(q)-mu| <= T`, cách cập nhật trung bình `mu`, số pixel `n`, lân cận 4 hướng.
- Cài đặt/kiểm tra hai thứ tự `stack` (LIFO) và `queue` (FIFO), hỗ trợ một hoặc nhiều seed.
- So sánh Stack với Queue; quét T, theo dõi pixel chưa gán, số vùng, diện tích vùng lớn nhất hoặc hiện tượng rò rỉ.
- Phối hợp thành viên 5 để so RG và SRG trên cùng ảnh, cùng seed.
- Bàn giao ảnh phân vùng, bảng/đồ thị quét T và số pixel khác nhau.

### Thành viên 5 — Seeded Region Growing (SRG)
- Nắm delta theo trung bình vùng, min-heap, boundary mask, stale `lazy`/`frozen`, tie-breaking.
- Kiểm tra phân hoạch đầy đủ, số vùng bằng số seed và thử đảo thứ tự seed.
- So sánh `lazy` với `frozen` trên ảnh gradient; thử tie `insertion/raster/random`, connectivity 4/8.
- Tạo frame phát triển vùng và bản đồ `seed_sensitivity`; phối hợp thành viên 4 so sánh RG/SRG cùng dữ liệu.
- Bàn giao nhãn cuối, boundary mask, frame, bản đồ độ nhạy và bảng số liệu.

## 3. Quy trình phối hợp

1. Thành viên 1 hoàn tất và thống nhất giao diện Harris trước; thành viên 2 và 3 dùng cùng đầu ra/ảnh thử khi so sánh.
2. Thành viên 4 và 5 thống nhất ảnh, seed, connectivity và cách tính số pixel khác nhau trước khi chạy RG/SRG.
3. Mỗi thành viên chạy notebook riêng từ đầu đến cuối và lưu output.
4. Cả nhóm chạy `notebooks/06_integrated_comparison.ipynb`, sau đó chạy `pytest -q` và `python scripts/run_all.py` tại thư mục gốc.
5. Mỗi thành viên phải giải thích được cơ chế thuật toán và kết quả thí nghiệm của phần mình, không chỉ biết bấm Run.

## 4. Checklist bàn giao cho từng thành viên

- [ ] Hiểu và trình bày được lý thuyết đúng với phần được giao.
- [ ] Kiểm tra hàm và định dạng đầu ra theo đề.
- [ ] Có ảnh minh họa kết quả.
- [ ] Có ít nhất một thí nghiệm so sánh định lượng.
- [ ] Có quét tham số và biểu đồ/đường cong phù hợp.
- [ ] Giải thích hiện tượng bằng bước tính toán cụ thể.
- [ ] Notebook chạy từ đầu đến cuối, output được lưu.

## 5. Lưu ý quan trọng về quy định AI trong đề

Tài liệu đề bài ghi rõ các cell cài đặt thuật toán thuộc phần “KHÔNG dùng AI” phải do sinh viên tự viết, kể cả sửa lỗi/viết lại bằng AI. AI chỉ được hỗ trợ một số công việc ở phần thí nghiệm và phải khai báo nguyên văn prompt ngay phía trên cell có sử dụng. Vì vậy project này nên được dùng làm tài liệu tham khảo/học tập và khung tổ chức; nhóm cần tự viết hoặc tự xác minh lại phần cài đặt theo đúng quy định của giảng viên trước khi nộp. Không nên nộp nguyên xi code do AI tạo.
