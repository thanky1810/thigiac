"""Region Growing cổ điển: mở rộng theo stack (LIFO) hoặc queue (FIFO)."""
from __future__ import annotations
from collections import deque
import numpy as np

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def region_growing_classic(img, seeds, T: float, order: str = "stack") -> np.ndarray:
    """Phân vùng ảnh bằng tiêu chí ||I(q)-mu|| <= T.

    seeds: danh sách tọa độ (row, col), một seed khởi tạo một vùng.
    order='stack' dùng LIFO; order='queue' dùng FIFO.
    Pixel không được kết nạp giữ nhãn -1. Thuật toán dùng lân cận 4.
    Hỗ trợ ảnh xám (H,W) và ảnh màu (H,W,C).
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError("img phải là ảnh xám (H,W) hoặc ảnh màu (H,W,C).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("img chứa NaN hoặc infinity.")
    if T < 0:
        raise ValueError("T phải >= 0.")
    if order not in ("stack", "queue"):
        raise ValueError("order chỉ nhận 'stack' hoặc 'queue'.")
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("Phải cung cấp ít nhất một seed.")

    h, w = arr.shape[:2]
    labels = np.full((h, w), -1, dtype=np.int32)
    sums: list[np.ndarray | float] = []
    counts: list[int] = []
    normalized_seeds: list[tuple[int, int]] = []

    for seed in seed_list:
        if len(seed) != 2:
            raise ValueError("Mỗi seed phải có dạng (row, col).")
        y, x = int(seed[0]), int(seed[1])
        if not (0 <= y < h and 0 <= x < w):
            raise ValueError(f"Seed {(y, x)} nằm ngoài ảnh.")
        if labels[y, x] != -1:
            raise ValueError("Các seed không được trùng tọa độ.")
        region_id = len(normalized_seeds)
        labels[y, x] = region_id
        normalized_seeds.append((y, x))
        sums.append(np.array(arr[y, x], dtype=float, copy=True))
        counts.append(1)

    frontier = deque((y, x, rid) for rid, (y, x) in enumerate(normalized_seeds))
    while frontier:
        y, x, rid = frontier.pop() if order == "stack" else frontier.popleft()
        mean = sums[rid] / counts[rid]
        for dy, dx in _NEIGHBORS_4:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or labels[ny, nx] != -1:
                continue
            diff = float(np.linalg.norm(arr[ny, nx] - mean))
            if diff <= T:
                labels[ny, nx] = rid
                sums[rid] = sums[rid] + arr[ny, nx]
                counts[rid] += 1
                frontier.append((ny, nx, rid))
    return labels
