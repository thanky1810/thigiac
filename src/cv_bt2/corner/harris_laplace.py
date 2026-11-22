"""Biến thể Harris-Laplace với chọn tỉ lệ bằng cực trị LoG."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .harris_classic import _as_gray_float, harris_response, nms_2d


def scale_pyramid(sigma0: float = 1.2, xi: float = 1.2, n_levels: int = 10) -> np.ndarray:
    """Tạo dãy sigma_n = sigma0 * xi**n."""
    if sigma0 <= 0:
        raise ValueError("sigma0 phải > 0.")
    if xi <= 1:
        raise ValueError("xi phải > 1 để tháp tỉ lệ tăng dần.")
    if int(n_levels) != n_levels or n_levels < 3:
        raise ValueError("n_levels phải là số nguyên >= 3.")
    return sigma0 * np.power(float(xi), np.arange(int(n_levels), dtype=float))


def log_normalized(img: np.ndarray, sigma: float) -> np.ndarray:
    """Tính |LoG| chuẩn hóa theo tỉ lệ: sigma**2 * |Lxx + Lyy|.

    Dùng đạo hàm Gaussian bậc hai trực tiếp, không gọi detector đặc trưng dựng sẵn.
    """
    arr = _as_gray_float(img)
    if sigma <= 0:
        raise ValueError("sigma phải > 0.")
    Lxx = gaussian_filter(arr, sigma=float(sigma), order=(0, 2), mode="reflect")
    Lyy = gaussian_filter(arr, sigma=float(sigma), order=(2, 0), mode="reflect")
    return (float(sigma) ** 2) * np.abs(Lxx + Lyy)


def harris_laplace(
    img: np.ndarray,
    sigmas: np.ndarray,
    k: float = 0.04,
    thresh_rel: float = 0.01,
    log_thresh: float = 0.0,
    nms_window: int = 3,
) -> np.ndarray:
    """Harris-Laplace (Mikolajczyk & Schmid).

    Bước 1: tìm cực đại Harris trong mặt phẳng ảnh tại từng sigma.
    Bước 2: chỉ giữ ứng viên có LoG chuẩn hóa là cực đại theo trục tỉ lệ.

    Returns ndarray (N, 4): [row, col, sigma, harris_response].
    """
    arr = _as_gray_float(img)
    scales = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    if scales.size < 3 or np.any(scales <= 0) or np.any(np.diff(scales) <= 0):
        raise ValueError("sigmas phải chứa ít nhất 3 giá trị dương, tăng nghiêm ngặt.")
    if thresh_rel < 0 or log_thresh < 0:
        raise ValueError("thresh_rel và log_thresh phải >= 0.")

    log_maps = [log_normalized(arr, s) for s in scales]
    found: list[tuple[float, float, float, float]] = []

    for n in range(1, len(scales) - 1):
        sigma = float(scales[n])
        R = harris_response(arr, sigma_i=sigma, sigma_d=0.7 * sigma, k=k)
        max_R = float(np.max(R))
        if max_R <= 0:
            continue
        candidates = nms_2d(R, window=nms_window)
        candidates &= (R > 0) & (R >= thresh_rel * max_R)
        scale_max = (log_maps[n] > log_maps[n - 1])
        scale_max &= (log_maps[n] > log_maps[n + 1])
        scale_max &= (log_maps[n] >= log_thresh)
        rows, cols = np.where(candidates & scale_max)
        for r, c in zip(rows, cols):
            found.append((float(r), float(c), sigma, float(R[r, c])))

    if not found:
        return np.empty((0, 4), dtype=np.float64)
    points = np.asarray(found, dtype=np.float64)
    order = np.lexsort((points[:, 1], points[:, 0], -points[:, 3]))
    return points[order]
