"""Harris corner detector tự cài đặt theo tài liệu Bài tập 2."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _as_gray_float(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Ảnh đầu vào phải là ảnh xám 2 chiều.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Ảnh chứa NaN hoặc infinity.")
    return arr


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Làm mờ Gauss; sigma là độ lệch chuẩn theo đơn vị pixel."""
    arr = _as_gray_float(img)
    if sigma < 0:
        raise ValueError("sigma phải >= 0.")
    if sigma == 0:
        return arr.copy()
    return gaussian_filter(arr, sigma=float(sigma), mode="reflect")


def structure_tensor(
    img: np.ndarray, sigma_i: float, sigma_d: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính Mxx, Mxy, Myy theo công thức (1) của tài liệu.

    sigma_d: làm trơn trước khi lấy gradient.
    sigma_i: làm trơn/tích phân các tích gradient.
    Cả ba thành phần được nhân sigma_d**2 (chuẩn hóa theo tỉ lệ).
    """
    arr = _as_gray_float(img)
    if sigma_i < 0 or sigma_d < 0:
        raise ValueError("sigma_i và sigma_d phải >= 0.")

    smooth = gaussian_blur(arr, sigma_d)
    # np.gradient trả về đạo hàm theo hàng (y) rồi theo cột (x).
    Iy, Ix = np.gradient(smooth)
    scale = float(sigma_d) ** 2

    Mxx = gaussian_blur(Ix * Ix, sigma_i) * scale
    Mxy = gaussian_blur(Ix * Iy, sigma_i) * scale
    Myy = gaussian_blur(Iy * Iy, sigma_i) * scale
    return Mxx, Mxy, Myy


def harris_response(
    img: np.ndarray, sigma_i: float, sigma_d: float, k: float = 0.04
) -> np.ndarray:
    """Tính R = det(M) - k * trace(M)**2."""
    if not (0 <= k < 1):
        raise ValueError("k phải thuộc [0, 1).")
    Mxx, Mxy, Myy = structure_tensor(img, sigma_i, sigma_d)
    det_M = Mxx * Myy - Mxy * Mxy
    trace_M = Mxx + Myy
    return det_M - float(k) * trace_M * trace_M


def nms_2d(response: np.ndarray, window: int = 3) -> np.ndarray:
    """Mặt nạ cực đại địa phương trong cửa sổ ``window`` x ``window``.

    NMS được cài đặt bằng NumPy, không gọi detector/NMS dựng sẵn. Với các
    pixel đồng hạng thuộc cùng plateau, quy tắc row-major giữ lại một đại
    diện xác định để tránh nhiều điểm lặp quanh một góc.
    """
    R = np.asarray(response, dtype=np.float64)
    if R.ndim != 2:
        raise ValueError("response phải là mảng 2 chiều.")
    if window < 1 or window % 2 == 0:
        raise ValueError("window phải là số lẻ dương.")
    if not np.all(np.isfinite(R)):
        raise ValueError("response chứa NaN hoặc infinity.")

    h, w = R.shape
    radius = window // 2
    padded = np.pad(R, radius, mode="constant", constant_values=-np.inf)
    local_max = np.full((h, w), -np.inf, dtype=np.float64)
    for dy in range(window):
        for dx in range(window):
            local_max = np.maximum(local_max, padded[dy:dy + h, dx:dx + w])
    candidates = R == local_max
    keep = candidates.copy()

    # Nếu một pixel có ứng viên đồng hạng đứng trước nó trong cửa sổ NMS,
    # loại pixel hiện tại. Dùng candidates gốc (không phải keep) để loại cả
    # chuỗi plateau, chỉ giữ một đại diện row-major.
    for dy in range(-radius, 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx >= 0:
                continue
            r0, r1 = max(0, -dy), min(h, h - dy)
            c0, c1 = max(0, -dx), min(w, w - dx)
            if r0 >= r1 or c0 >= c1:
                continue
            current = (slice(r0, r1), slice(c0, c1))
            earlier = (slice(r0 + dy, r1 + dy), slice(c0 + dx, c1 + dx))
            tied_prior = candidates[earlier] & (R[earlier] == R[current])
            keep[current] &= ~tied_prior
    return keep


def harris_classic(
    img: np.ndarray,
    sigma_i: float = 2.0,
    sigma_d: float = 1.4,
    k: float = 0.04,
    thresh_rel: float = 0.01,
    nms_window: int = 3,
) -> np.ndarray:
    """Phát hiện góc Harris một tỉ lệ.

    Returns
    -------
    points : ndarray, shape (N, 3)
        Mỗi hàng [row, col, response], sắp giảm dần theo response.
    """
    arr = _as_gray_float(img)
    if thresh_rel < 0:
        raise ValueError("thresh_rel phải >= 0.")
    R = harris_response(arr, sigma_i, sigma_d, k=k)
    max_R = float(np.max(R)) if R.size else 0.0
    if max_R <= 0:
        return np.empty((0, 3), dtype=np.float64)

    mask = nms_2d(R, window=nms_window)
    mask &= (R > 0) & (R >= thresh_rel * max_R)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    points = np.column_stack((rows, cols, R[rows, cols])).astype(np.float64)
    order = np.lexsort((points[:, 1], points[:, 0], -points[:, 2]))
    return points[order]
