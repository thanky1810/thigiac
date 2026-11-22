"""Các phép đo/kiểm chứng cho phân vùng ảnh."""
from __future__ import annotations
import numpy as np



def _maximum_assignment_sum(weights: np.ndarray) -> int:
    """Tổng trọng số ghép cặp lớn nhất bằng Hungarian O(n^3), chỉ dùng NumPy/Python."""
    values = np.asarray(weights, dtype=np.int64)
    if values.ndim != 2:
        raise ValueError("weights phải là ma trận 2 chiều.")
    rows, cols = values.shape
    if rows == 0 or cols == 0:
        return 0
    n = max(rows, cols)
    padded = np.zeros((n, n), dtype=np.int64)
    padded[:rows, :cols] = values
    cost = -padded.astype(np.float64)

    # Thuật toán Hungarian dạng potential (1-indexed), tối thiểu hóa cost.
    u = np.zeros(n + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    total = 0
    for j in range(1, n + 1):
        assigned_row = int(p[j]) - 1
        assigned_col = j - 1
        if 0 <= assigned_row < rows and assigned_col < cols:
            total += int(values[assigned_row, assigned_col])
    return total

def count_regions(labels: np.ndarray) -> int:
    labels = np.asarray(labels)
    valid = labels[labels >= 0]
    return int(np.unique(valid).size)


def check_full_partition(labels: np.ndarray) -> bool:
    return bool(np.all(np.asarray(labels) >= 0))


def check_region_count(labels: np.ndarray, seeds) -> bool:
    return count_regions(labels) == len(seeds)


def changed_pixel_count(labels_a: np.ndarray, labels_b: np.ndarray) -> int:
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("Hai bản đồ nhãn phải cùng kích thước.")
    return int(np.count_nonzero(a != b))


def compare_partition(labels_a: np.ndarray, labels_b: np.ndarray) -> int:
    """Số pixel không khớp sau khi ánh xạ nhãn tối ưu (Hungarian).

    Không bị ảnh hưởng bởi việc hai thuật toán đặt ID vùng khác nhau.
    Chỉ so trên pixel được cả hai kết quả gán nhãn; caller nên báo cáo riêng
    số pixel chưa gán nếu một thuật toán có nhãn -1.
    """
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("Hai bản đồ nhãn phải cùng kích thước.")
    common = (a >= 0) & (b >= 0)
    if not np.any(common):
        return 0
    av, bv = a[common].ravel(), b[common].ravel()
    ua, ub = np.unique(av), np.unique(bv)
    overlap = np.zeros((len(ua), len(ub)), dtype=np.int64)
    amap = {v: i for i, v in enumerate(ua)}
    bmap = {v: j for j, v in enumerate(ub)}
    for x, y in zip(av, bv):
        overlap[amap[x], bmap[y]] += 1
    matched = _maximum_assignment_sum(overlap)
    return int(len(av) - matched)


def partition_difference(labels_a: np.ndarray, labels_b: np.ndarray) -> dict:
    """Tóm tắt so sánh hai phân vùng, gồm vùng chung và pixel chưa gán."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("Hai bản đồ nhãn phải cùng kích thước.")
    common = (a >= 0) & (b >= 0)
    total_common = int(common.sum())
    changed = compare_partition(a, b)
    return {
        "common_labeled_pixels": total_common,
        "different_partition_pixels": changed,
        "different_percent_of_common": (100.0 * changed / total_common) if total_common else 0.0,
        "unassigned_a": int(np.count_nonzero(a < 0)),
        "unassigned_b": int(np.count_nonzero(b < 0)),
    }
