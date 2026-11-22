"""Các hàm hỗ trợ hiển thị và lưu kết quả thực nghiệm."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def colorize_labels(labels: np.ndarray, cmap: str = "tab20"):
    """Trả về colormap và giới hạn hiển thị ổn định cho bản đồ nhãn."""
    arr = np.asarray(labels)
    n = max(int(np.max(arr)) + 1, 1)
    cm = plt.get_cmap(cmap, n)
    return cm, -0.5, n - 0.5


def plot_labels(labels, ax=None, title="Bản đồ phân vùng", cmap="tab20"):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    cm, vmin, vmax = colorize_labels(labels, cmap)
    im = ax.imshow(np.ma.masked_where(np.asarray(labels) < 0, labels), cmap=cm,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    return im, ax


def plot_corners(img, points, ax=None, title="Các điểm góc"):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(img, cmap="gray")
    pts = np.asarray(points)
    if pts.size:
        ax.scatter(pts[:, 1], pts[:, 0], s=18, c="red", marker="x")
    ax.set_title(title)
    ax.axis("off")
    return ax


def boundary_from_labels(labels, connectivity=4):
    """Boundary mask dùng trong trực quan hóa; khuyến nghị dùng mask từ SRG."""
    arr = np.asarray(labels)
    h, w = arr.shape
    boundary = np.zeros((h, w), dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        diff = arr[y0:y1, x0:x1] != arr[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        boundary[y0:y1, x0:x1] |= diff
        boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= diff
    return boundary


def save_figure(path, dpi=180):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
