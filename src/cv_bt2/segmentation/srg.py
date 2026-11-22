"""Seeded Region Growing (SRG) với heapq, lazy/frozen và tie-breaking.

Độ ưu tiên delta được hiểu là độ tương phản không âm:
    delta(x, A_i) = ||I(x) - mean(A_i)||_2.
Với ảnh xám, đây là trị tuyệt đối của hiệu cường độ; với ảnh màu là chuẩn L2.
"""
from __future__ import annotations

import heapq
import itertools
from typing import Callable
import numpy as np

_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),             (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _prepare(img, seeds, connectivity, stale, tie):
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError("img phải là ảnh xám (H,W) hoặc ảnh màu (H,W,C).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("img chứa NaN hoặc infinity.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity chỉ nhận 4 hoặc 8.")
    if stale not in ("lazy", "frozen"):
        raise ValueError("stale chỉ nhận 'lazy' hoặc 'frozen'.")
    if tie not in ("insertion", "raster", "random"):
        raise ValueError("tie chỉ nhận 'insertion', 'raster' hoặc 'random'.")
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("Phải cung cấp ít nhất một seed.")
    h, w = arr.shape[:2]
    normalized = []
    used = set()
    for seed in seed_list:
        if len(seed) != 2:
            raise ValueError("Mỗi seed phải có dạng (row, col).")
        y, x = int(seed[0]), int(seed[1])
        if not (0 <= y < h and 0 <= x < w):
            raise ValueError(f"Seed {(y, x)} nằm ngoài ảnh.")
        if (y, x) in used:
            raise ValueError("Các seed không được trùng tọa độ.")
        used.add((y, x))
        normalized.append((y, x))
    return arr, normalized


def _boundary_mask(labels: np.ndarray, connectivity: int) -> np.ndarray:
    """Đánh dấu hai phía của mọi cặp pixel lân cận có nhãn khác nhau."""
    h, w = labels.shape
    mask = np.zeros((h, w), dtype=bool)
    offsets = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        a = labels[y0:y1, x0:x1]
        b = labels[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        diff = a != b
        mask[y0:y1, x0:x1] |= diff
        mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= diff
    return mask


def _srg_run(
    img,
    seeds,
    connectivity=4,
    stale="lazy",
    tie="insertion",
    random_state=42,
    frame_targets=None,
):
    arr, seed_list = _prepare(img, seeds, connectivity, stale, tie)
    h, w = arr.shape[:2]
    neighbors = _NEIGHBORS_4 if connectivity == 4 else _NEIGHBORS_8
    n_regions = len(seed_list)

    labels = np.full((h, w), -1, dtype=np.int32)
    region_sum = np.zeros((n_regions,) + arr.shape[2:], dtype=np.float64)
    region_count = np.ones(n_regions, dtype=np.int64)
    seed_values = np.empty((n_regions,) + arr.shape[2:], dtype=np.float64)
    for rid, (y, x) in enumerate(seed_list):
        labels[y, x] = rid
        region_sum[rid] = arr[y, x]
        seed_values[rid] = arr[y, x]

    rng = np.random.default_rng(random_state)
    counter = itertools.count()
    heap = []
    versions = np.zeros((h, w), dtype=np.int64)
    snapshots = []
    targets = list(frame_targets or [])
    target_index = 0

    def neighbor_region_ids(y, x):
        ids = set()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                rid = int(labels[ny, nx])
                if rid >= 0:
                    ids.add(rid)
        return sorted(ids)

    def score_for_region(y, x, rid):
        if stale == "lazy":
            mean = region_sum[rid] / region_count[rid]
        else:
            mean = seed_values[rid]
        return float(np.linalg.norm(arr[y, x] - mean))

    def best_adjacent_region(y, x):
        ids = neighbor_region_ids(y, x)
        if not ids:
            return None, np.inf
        scored = [(score_for_region(y, x, rid), rid) for rid in ids]
        # Region ID là quy tắc phụ tất định khi một pixel kề nhiều vùng có delta bằng nhau.
        delta, rid = min(scored, key=lambda item: (item[0], item[1]))
        return rid, delta

    def push_candidate(y, x):
        if labels[y, x] >= 0:
            return
        rid, delta = best_adjacent_region(y, x)
        if rid is None:
            return
        versions[y, x] += 1
        version = int(versions[y, x])
        seq = next(counter)
        if tie == "insertion":
            tie_key = seq
        elif tie == "raster":
            tie_key = y * w + x
        else:
            tie_key = float(rng.random())
        heapq.heappush(heap, (delta, tie_key, seq, version, y, x, rid))

    # Khởi tạo biên từ tất cả seed.
    for y, x in seed_list:
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] < 0:
                push_candidate(ny, nx)

    processed = n_regions
    total = h * w
    while heap:
        stored_delta, _tie_key, _seq, version, y, x, stored_rid = heapq.heappop(heap)
        if labels[y, x] >= 0 or version != versions[y, x]:
            continue

        current_rid, current_delta = best_adjacent_region(y, x)
        if current_rid is None:
            continue

        if stale == "lazy":
            changed = (current_rid != stored_rid) or not np.isclose(
                current_delta, stored_delta, rtol=1e-12, atol=1e-12
            )
            if changed:
                push_candidate(y, x)
                continue

        # Kết nạp pixel vào vùng có delta nhỏ nhất hiện tại.
        rid = current_rid
        labels[y, x] = rid
        region_sum[rid] += arr[y, x]
        region_count[rid] += 1
        processed += 1

        # Cập nhật các pixel biên mới lộ ra.
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and labels[ny, nx] < 0:
                push_candidate(ny, nx)

        while target_index < len(targets) and processed >= targets[target_index]:
            snapshots.append(labels.copy())
            target_index += 1

    # Với lưới ảnh liên thông và ít nhất một seed, mọi pixel phải được gán.
    # Nếu còn pixel chưa nhãn, đây là tình huống bất thường do lỗi logic/đầu vào.
    if np.any(labels < 0):
        raise RuntimeError("SRG kết thúc nhưng vẫn còn pixel chưa được gán nhãn.")

    while target_index < len(targets):
        snapshots.append(labels.copy())
        target_index += 1
    return labels, _boundary_mask(labels, connectivity), snapshots


def srg(
    img,
    seeds,
    connectivity: int = 4,
    stale: str = "lazy",
    tie: str = "insertion",
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded Region Growing dùng min-heap.

    Parameters
    ----------
    connectivity : 4 hoặc 8
    stale : 'lazy' (cập nhật delta theo mean hiện tại) hoặc 'frozen'
            (delta theo cường độ seed ban đầu).
    tie : 'insertion', 'raster' hoặc 'random'. Random có thể tái lập qua random_state.

    Returns
    -------
    labels : int32 (H,W), mọi pixel có nhãn >= 0.
    boundary_mask : bool (H,W), đánh dấu cả hai phía của ranh giới vùng.
    """
    labels, boundary, _ = _srg_run(
        img, seeds, connectivity, stale, tie, random_state=random_state
    )
    return labels, boundary


def srg_step_frames(
    img,
    seeds,
    n_frames: int = 8,
    **kwargs,
) -> list[np.ndarray]:
    """Trả về n_frames ảnh nhãn ở các mốc tiến trình, frame cuối = srg()."""
    if int(n_frames) != n_frames or n_frames < 1:
        raise ValueError("n_frames phải là số nguyên >= 1.")
    arr = np.asarray(img)
    total = arr.shape[0] * arr.shape[1]
    targets = np.ceil(np.linspace(1, total, int(n_frames))).astype(int).tolist()
    labels, _boundary, frames = _srg_run(
        img,
        seeds,
        connectivity=kwargs.pop("connectivity", 4),
        stale=kwargs.pop("stale", "lazy"),
        tie=kwargs.pop("tie", "insertion"),
        random_state=kwargs.pop("random_state", 42),
        frame_targets=targets,
    )
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Tham số không được hỗ trợ: {unknown}")
    # Bảo đảm đúng số frame ngay cả khi một số target làm tròn trùng nhau.
    while len(frames) < int(n_frames):
        frames.append(labels.copy())
    return frames[: int(n_frames)]


def seed_sensitivity(
    img,
    seeds,
    segment_fn: Callable,
    n_trials: int = 20,
    jitter: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Bản đồ tần suất pixel đổi nhãn khi seed bị dịch chuyển.

    Mỗi pixel lưu tỉ lệ lần thử mà nhãn khác kết quả baseline, trong [0,1].
    Các seed dịch chuyển ra ngoài ảnh được clip về biên; seed trùng nhau bị bỏ qua.
    """
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError("img phải có 2 hoặc 3 chiều.")
    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("Danh sách seeds không được rỗng.")
    if n_trials < 1 or int(n_trials) != n_trials:
        raise ValueError("n_trials phải là số nguyên >= 1.")
    if jitter < 0:
        raise ValueError("jitter phải >= 0.")

    baseline_result = segment_fn(img, seed_list)
    baseline = baseline_result[0] if isinstance(baseline_result, tuple) else baseline_result
    baseline = np.asarray(baseline)
    h, w = arr.shape[:2]
    if baseline.shape != (h, w):
        raise ValueError("Kích thước kết quả phân đoạn không hợp lệ.")

    rng = np.random.default_rng(random_state)
    changes = np.zeros((h, w), dtype=np.float64)
    successes = 0
    attempts = 0
    max_attempts = max(int(n_trials) * 30, int(n_trials))
    while successes < int(n_trials) and attempts < max_attempts:
        attempts += 1
        shifted = []
        for y, x in seed_list:
            dy = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            dx = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            shifted.append((int(np.clip(int(y) + dy, 0, h - 1)),
                            int(np.clip(int(x) + dx, 0, w - 1))))
        if len(set(shifted)) != len(shifted):
            continue
        result = segment_fn(img, shifted)
        current = result[0] if isinstance(result, tuple) else result
        current = np.asarray(current)
        if current.shape != baseline.shape:
            raise ValueError("segment_fn trả về nhãn có kích thước không đồng nhất.")
        changes += (current != baseline)
        successes += 1

    if successes == 0:
        raise RuntimeError("Không tạo được trial hợp lệ; hãy tăng kích thước ảnh hoặc giảm số seed.")
    if successes < int(n_trials):
        raise RuntimeError(
            f"Chỉ tạo được {successes}/{n_trials} trial hợp lệ; hãy giảm jitter hoặc số seed."
        )
    return (changes / successes).astype(np.float32)
