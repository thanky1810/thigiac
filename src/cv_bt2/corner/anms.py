"""Adaptive Non-Maximal Suppression và phép đo độ phủ lưới."""
from __future__ import annotations

import numpy as np


def _validate_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points phải có dạng (N, >=3): row, col, response.")
    if not np.all(np.isfinite(pts[:, :3])):
        raise ValueError("points chứa giá trị không hữu hạn.")
    return pts[:, :3]


def top_n(points: np.ndarray, n_keep: int) -> np.ndarray:
    """Đường cơ sở: giữ n_keep điểm có Harris response cao nhất."""
    pts = _validate_points(points)
    if n_keep < 0:
        raise ValueError("n_keep phải >= 0.")
    if n_keep == 0 or len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)
    order = np.lexsort((pts[:, 1], pts[:, 0], -pts[:, 2]))
    return pts[order[: min(int(n_keep), len(pts))]].copy()


def anms(points: np.ndarray, n_keep: int, c_robust: float = 0.9) -> np.ndarray:
    """Chọn điểm theo bán kính ức chế ANMS.

    Với điểm i, r_i là khoảng cách đến điểm gần nhất j thỏa
    f_i < c_robust * f_j. Điểm mạnh nhất có r_i = vô cực.
    Độ phức tạp O(N^2), phù hợp với tập ứng viên Harris cỡ vừa của bài tập.
    """
    pts = _validate_points(points)
    if n_keep < 0:
        raise ValueError("n_keep phải >= 0.")
    if not (0 < c_robust <= 1):
        raise ValueError("c_robust phải thuộc (0, 1].")
    if n_keep == 0 or len(pts) == 0:
        return np.empty((0, 3), dtype=np.float64)

    n = len(pts)
    radii = np.full(n, np.inf, dtype=np.float64)
    coords = pts[:, :2]
    responses = pts[:, 2]

    # Chia block để tránh tạo ma trận khoảng cách N x N quá lớn cùng lúc.
    block_size = 512
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        block = coords[start:stop]
        d2 = np.sum((block[:, None, :] - coords[None, :, :]) ** 2, axis=2)
        eligible = responses[None, :] > (responses[start:stop, None] / c_robust)
        d2[~eligible] = np.inf
        radii[start:stop] = np.sqrt(np.min(d2, axis=1))

    # Ưu tiên bán kính lớn; nếu bằng nhau thì response lớn hơn, sau đó tọa độ.
    order = np.lexsort((pts[:, 1], pts[:, 0], -responses, -radii))
    return pts[order[: min(int(n_keep), n)]].copy()


def grid_occupancy(
    points: np.ndarray,
    img_shape: tuple[int, int],
    grid_size: int | tuple[int, int] = 8,
) -> float:
    """Tỉ lệ ô lưới có ít nhất một điểm, giá trị trong [0, 1].

    grid_size là số ô theo mỗi chiều (g_rows, g_cols), không phải kích thước pixel.
    """
    pts = _validate_points(points)
    height, width = map(int, img_shape[:2])
    if height <= 0 or width <= 0:
        raise ValueError("img_shape phải có kích thước dương.")
    if isinstance(grid_size, int):
        grid_rows = grid_cols = grid_size
    else:
        grid_rows, grid_cols = map(int, grid_size)
    if grid_rows <= 0 or grid_cols <= 0:
        raise ValueError("grid_size phải dương.")
    if len(pts) == 0:
        return 0.0

    rows = np.floor(pts[:, 0] / height * grid_rows).astype(int)
    cols = np.floor(pts[:, 1] / width * grid_cols).astype(int)
    valid = (rows >= 0) & (rows < grid_rows) & (cols >= 0) & (cols < grid_cols)
    occupied = set(zip(rows[valid].tolist(), cols[valid].tolist()))
    return float(len(occupied) / (grid_rows * grid_cols))
