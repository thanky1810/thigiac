"""Hàm tiện ích cho thí nghiệm tổng hợp; thuật toán nằm trong các module riêng."""
from __future__ import annotations
import numpy as np
from .corner import harris_classic, harris_laplace, anms, top_n, repeatability, grid_occupancy
from .segmentation import region_growing_classic, srg, seed_sensitivity
from .segmentation.metrics import partition_difference
from .data.synthetic import checkerboard_image, scale_image, anms_scene, segmentation_scene


def run_corner_scale_experiment():
    img_a = checkerboard_image((120, 120), squares=6)
    img_b = scale_image(img_a, 3.0, order=1)
    transform = lambda r, c: (3.0 * r, 3.0 * c)
    p_h_a = harris_classic(img_a)
    p_h_b = harris_classic(img_b)
    scales_a = np.geomspace(0.9, 7.5, 12)
    scales_b = np.geomspace(0.9, 22.5, 18)
    p_hl_a = harris_laplace(img_a, scales_a, log_thresh=0.0)
    p_hl_b = harris_laplace(img_b, scales_b, log_thresh=0.0)
    return {
        "images": (img_a, img_b),
        "harris": (p_h_a, p_h_b, repeatability(p_h_a, p_h_b, transform, img_b.shape)),
        "harris_laplace": (p_hl_a, p_hl_b, repeatability(p_hl_a, p_hl_b, transform, img_b.shape)),
    }


def run_anms_experiment(n_keep=120, c_robust=0.9):
    img = anms_scene()
    pts = harris_classic(img, sigma_i=1.5, sigma_d=1.0, thresh_rel=0.005)
    return {
        "image": img,
        "all_points": pts,
        "top_n": top_n(pts, n_keep),
        "anms": anms(pts, n_keep, c_robust),
        "top_occupancy": grid_occupancy(top_n(pts, n_keep), img.shape, 8),
        "anms_occupancy": grid_occupancy(anms(pts, n_keep, c_robust), img.shape, 8),
    }


def run_segmentation_experiment(T=25):
    img = segmentation_scene()
    seeds = [(60, 20), (60, 90), (60, 160)]
    stack = region_growing_classic(img, seeds, T=T, order="stack")
    queue = region_growing_classic(img, seeds, T=T, order="queue")
    labels_srg, boundary = srg(img, seeds, stale="lazy", tie="insertion")
    return {
        "image": img, "seeds": seeds,
        "stack": stack, "queue": queue, "srg": labels_srg, "boundary": boundary,
        "stack_srg": partition_difference(stack, labels_srg),
        "queue_srg": partition_difference(queue, labels_srg),
    }


def default_seed_sensitivity(img, seeds, n_trials=20, jitter=3):
    return seed_sensitivity(
        img, seeds,
        lambda image, seed_list: srg(image, seed_list, stale="lazy", tie="insertion"),
        n_trials=n_trials, jitter=jitter,
    )
