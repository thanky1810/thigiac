"""Các phép đo dùng chung cho bài toán phát hiện điểm góc."""
from __future__ import annotations
import numpy as np


def repeatability(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    transform,
    img_shape: tuple[int, int],
    eps: float = 3.0,
) -> float:
    """Tỉ lệ điểm ảnh A có điểm tương ứng trong ảnh B sau biến đổi.

    Một điểm A được tính là tìm lại được nếu tồn tại ít nhất một điểm B
    cách tọa độ chiếu không quá ``eps`` pixel. Tính theo từng block để tránh
    cấp phát ma trận khoảng cách quá lớn, chỉ dùng NumPy.
    """
    a = np.asarray(pts_a, dtype=float)
    b = np.asarray(pts_b, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2 or b.ndim != 2 or b.shape[1] < 2:
        raise ValueError("pts_a và pts_b phải có dạng (N, >=2).")
    if eps < 0:
        raise ValueError("eps phải >= 0.")
    if len(a) == 0:
        return 0.0
    mapped = np.asarray([transform(float(p[0]), float(p[1])) for p in a], dtype=float)
    h, w = img_shape[:2]
    visible = (
        (mapped[:, 0] >= 0) & (mapped[:, 0] < h) &
        (mapped[:, 1] >= 0) & (mapped[:, 1] < w)
    )
    mapped = mapped[visible]
    if len(mapped) == 0 or len(b) == 0:
        return 0.0

    matched = 0
    block_size = 256
    target = b[:, :2]
    eps2 = float(eps) ** 2
    for start in range(0, len(mapped), block_size):
        block = mapped[start:start + block_size]
        distances2 = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        matched += int(np.count_nonzero(np.min(distances2, axis=1) <= eps2))
    return float(matched / len(mapped))
