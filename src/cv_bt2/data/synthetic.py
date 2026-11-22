"""Bộ ảnh tổng hợp tái lập để notebook chạy độc lập, không cần ảnh tải ngoài."""
from __future__ import annotations
import numpy as np


def checkerboard_image(size=(128, 128), squares=8, noise=0.0, random_state=7):
    """Bàn cờ xám [0,1], thuận tiện kiểm tra các giao điểm Harris."""
    h, w = map(int, size)
    if h <= 0 or w <= 0 or squares < 2:
        raise ValueError("size phải dương và squares >= 2.")
    yy, xx = np.indices((h, w))
    board = ((yy * squares // h + xx * squares // w) % 2).astype(float)
    img = 0.15 + 0.70 * board
    if noise > 0:
        rng = np.random.default_rng(random_state)
        img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 1).astype(np.float64)


def _resize_bilinear(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Nội suy song tuyến tính 2D để không phụ thuộc hàm resize dựng sẵn."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Hàm resize hỗ trợ ảnh xám 2 chiều.")
    out_h, out_w = map(int, output_shape)
    in_h, in_w = arr.shape
    if min(out_h, out_w, in_h, in_w) <= 0:
        raise ValueError("Kích thước ảnh phải dương.")
    y = np.linspace(0, in_h - 1, out_h)
    x = np.linspace(0, in_w - 1, out_w)
    y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, in_h - 1); x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (y - y0)[:, None]; wx = (x - x0)[None, :]
    top = arr[y0[:, None], x0[None, :]] * (1 - wx) + arr[y0[:, None], x1[None, :]] * wx
    bottom = arr[y1[:, None], x0[None, :]] * (1 - wx) + arr[y1[:, None], x1[None, :]] * wx
    return top * (1 - wy) + bottom * wy


def scale_image(img, factor=3.0, order=1):
    """Phóng to/thu nhỏ ảnh bằng nội suy song tuyến tính NumPy.

    Tham số ``order`` được giữ để tương thích notebook cũ; project dùng order=1.
    """
    if factor <= 0:
        raise ValueError("factor phải > 0.")
    arr = np.asarray(img, dtype=float)
    shape = (max(1, int(round(arr.shape[0] * factor))),
             max(1, int(round(arr.shape[1] * factor))))
    return _resize_bilinear(arr, shape)


def rotate_image(img, angle_degrees: float, cval: float = 0.5) -> np.ndarray:
    """Xoay ảnh xám quanh tâm, giữ nguyên kích thước, nội suy song tuyến tính.

    Đây là phép biến đổi dùng cho thí nghiệm, không phải detector dựng sẵn.
    Góc dương có cùng quy ước với scipy.ndimage.rotate trên hệ tọa độ ảnh.
    """
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("rotate_image yêu cầu ảnh xám 2 chiều.")
    h, w = arr.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    theta = np.deg2rad(float(angle_degrees))
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    out_r, out_c = np.indices((h, w), dtype=np.float64)
    dx, dy = out_c - cx, out_r - cy
    src_c = cos_t * dx - sin_t * dy + cx
    src_r = sin_t * dx + cos_t * dy + cy

    valid = (src_r >= 0) & (src_r <= h - 1) & (src_c >= 0) & (src_c <= w - 1)
    r0 = np.floor(np.clip(src_r, 0, h - 1)).astype(int)
    c0 = np.floor(np.clip(src_c, 0, w - 1)).astype(int)
    r1 = np.minimum(r0 + 1, h - 1); c1 = np.minimum(c0 + 1, w - 1)
    wr = (src_r - r0); wc = (src_c - c0)
    top = arr[r0, c0] * (1 - wc) + arr[r0, c1] * wc
    bottom = arr[r1, c0] * (1 - wc) + arr[r1, c1] * wc
    out = top * (1 - wr) + bottom * wr
    out[~valid] = float(cval)
    return out


def anms_scene(size=(160, 240), random_state=11):
    """Cảnh có nhiều góc mạnh dày đặc ở nửa trái, nền ít cấu trúc ở nửa phải.

    Biên độ checkerboard tăng dần theo chiều ngang để response Harris biến thiên,
    làm rõ khác biệt giữa Top-N (ưu tiên response) và ANMS (ưu tiên độ cô lập).
    """
    h, w = map(int, size)
    img = np.full((h, w), 0.5, dtype=float)
    yy, xx = np.indices((h, w // 2))
    amplitude = 0.25 + 0.25 * (xx / max(w // 2 - 1, 1))
    pattern = ((yy // 8 + xx // 8) % 2).astype(float)
    img[:, :w // 2] = 0.5 + (2.0 * pattern - 1.0) * amplitude
    # Một số góc cường độ thấp ở nửa phải; chúng dễ bị Top-N bỏ qua.
    for y0, x0, rh, rw in [(20, 145, 25, 32), (95, 180, 35, 28), (45, 205, 18, 20)]:
        img[y0:y0 + rh, x0:x0 + rw] = 0.62
    rng = np.random.default_rng(random_state)
    img += rng.normal(0, 0.002, img.shape)
    return np.clip(img, 0, 1)


def segmentation_scene(size=(120, 180), random_state=42, noise=5.0):
    """Ba dải cường độ xám, có gradient nhẹ, gợn và nhiễu."""
    h, w = map(int, size)
    yy, xx = np.indices((h, w))
    img = np.zeros((h, w), dtype=float)
    cuts = [w // 3, 2 * w // 3]
    img[:, :cuts[0]] = 55
    img[:, cuts[0]:cuts[1]] = 125
    img[:, cuts[1]:] = 195
    img += (xx % max(cuts[0], 1)) * 0.22
    img += 5.0 * np.sin(yy / 9.0)
    rng = np.random.default_rng(random_state)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def gradient_scene(size=(80, 140), nonlinear=True):
    """Gradient ngang để làm lộ khác biệt Lazy và Frozen."""
    h, w = map(int, size)
    x = np.linspace(0, 1, w)
    if nonlinear:
        x = x ** 2
    return np.tile(255.0 * x, (h, 1)).astype(np.float64)


def flat_scene(size=(100, 160), value=128):
    """Ảnh phẳng, mọi ứng viên có cùng delta, dùng kiểm tra tie-breaking."""
    return np.full(tuple(map(int, size)), float(value), dtype=np.uint8)


def diagonal_scene(size=(100, 100), random_state=3):
    """Ảnh có dải chéo để so sánh lân cận 4 và 8."""
    h, w = map(int, size)
    yy, xx = np.indices((h, w))
    img = np.where(np.abs(yy - xx) < 7, 210.0, 35.0)
    img = np.where(np.abs(yy + xx - (w - 1)) < 6, 130.0, img)
    rng = np.random.default_rng(random_state)
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
