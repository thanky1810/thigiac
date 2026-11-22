import numpy as np

from cv_bt2.corner import (
    harris_classic, harris_laplace, scale_pyramid,
    top_n, anms, grid_occupancy, repeatability,
)
from cv_bt2.data.synthetic import (
    checkerboard_image, scale_image, anms_scene, segmentation_scene,
    gradient_scene, flat_scene,
)
from cv_bt2.segmentation import (
    region_growing_classic, srg, srg_step_frames, seed_sensitivity,
    count_regions, check_full_partition,
)
from cv_bt2.segmentation.metrics import partition_difference


def test_harris_finds_checkerboard_internal_corners():
    img = checkerboard_image((120, 120), squares=6)
    points = harris_classic(img)
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(points) >= 20
    assert np.all(points[:, 2] > 0)
    assert np.all(points[:-1, 2] >= points[1:, 2])


def test_harris_laplace_output_shape_and_repeatability():
    img_a = checkerboard_image((120, 120), squares=6)
    img_b = scale_image(img_a, 3.0)
    points_a = harris_laplace(img_a, scale_pyramid(0.25, 1.2, 10))
    points_b = harris_laplace(img_b, scale_pyramid(0.25, 1.2, 16))
    assert points_a.ndim == 2 and points_a.shape[1] == 4
    assert points_b.ndim == 2 and points_b.shape[1] == 4
    rep = repeatability(points_a, points_b, lambda r, c: (3*r, 3*c), img_b.shape, eps=4)
    assert 0.0 <= rep <= 1.0
    assert len(points_a) > 0 and len(points_b) > 0


def test_anms_returns_same_count_and_better_grid_coverage_for_demo_image():
    img = anms_scene()
    candidates = harris_classic(img, sigma_i=1.5, sigma_d=1.0, thresh_rel=0.002)
    top = top_n(candidates, 100)
    suppressed = anms(candidates, 100, c_robust=0.9)
    assert len(top) == len(suppressed) == 100
    assert grid_occupancy(suppressed, img.shape, 8) >= grid_occupancy(top, img.shape, 8)


def test_region_growing_and_srg_partition_properties():
    img = segmentation_scene()
    seeds = [(60, 20), (60, 90), (60, 160)]
    labels_stack = region_growing_classic(img, seeds, 25, order="stack")
    labels_queue = region_growing_classic(img, seeds, 25, order="queue")
    labels_srg, boundary = srg(img, seeds)
    assert labels_stack.dtype == np.int32
    assert labels_queue.dtype == np.int32
    assert check_full_partition(labels_srg)
    assert count_regions(labels_srg) == len(seeds)
    assert boundary.dtype == bool and boundary.shape == img.shape
    assert partition_difference(labels_stack, labels_srg)["unassigned_b"] == 0


def test_srg_frames_last_matches_final_segmentation():
    img = segmentation_scene(size=(45, 60))
    seeds = [(22, 8), (22, 30), (22, 52)]
    final, _ = srg(img, seeds)
    frames = srg_step_frames(img, seeds, n_frames=6)
    assert len(frames) == 6
    assert np.array_equal(frames[-1], final)


def test_srg_lazy_frozen_and_tie_options_are_operational():
    grad = gradient_scene((40, 70), nonlinear=True)
    seeds = [(20, 3), (20, 66)]
    lazy, _ = srg(grad, seeds, stale="lazy")
    frozen, _ = srg(grad, seeds, stale="frozen")
    assert np.count_nonzero(lazy != frozen) > 0

    flat = flat_scene((40, 70))
    flat_seeds = [(20, 10), (20, 60)]
    insertion, _ = srg(flat, flat_seeds, tie="insertion")
    raster, _ = srg(flat, flat_seeds, tie="raster")
    random_a, _ = srg(flat, flat_seeds, tie="random", random_state=42)
    random_b, _ = srg(flat, flat_seeds, tie="random", random_state=42)
    assert np.count_nonzero(insertion != raster) > 0
    assert np.array_equal(random_a, random_b)


def test_seed_sensitivity_map_range():
    img = gradient_scene((40, 70), nonlinear=True)
    seeds = [(20, 8), (20, 62)]
    freq = seed_sensitivity(img, seeds, lambda a, s: srg(a, s), n_trials=4, jitter=2)
    assert freq.shape == img.shape
    assert np.all((freq >= 0) & (freq <= 1))
