"""Tests for dataset preprocessing helpers and the processed-file loader (PROJECT_SPEC Phase 2)."""

import numpy as np
import pytest
import torch

from phosphene_lab.datasets import (
    OBSTACLE_CLASSES,
    load_dataset,
    obstacle_label,
    square_crop_box,
)

# --- obstacle labels from depth -------------------------------------------------------


def _far(h=60, w=90):
    return np.full((h, w), 5.0, dtype=np.float32)


def test_open_space_has_no_obstacle():
    assert OBSTACLE_CLASSES[obstacle_label(_far())] == "none"


@pytest.mark.parametrize(
    ("cols", "name"), [((0, 30), "left"), ((30, 60), "center"), ((60, 90), "right")]
)
def test_near_blob_is_labelled_by_its_third(cols, name):
    depth = _far()
    depth[30:, cols[0] : cols[1]] = 0.6
    assert OBSTACLE_CLASSES[obstacle_label(depth)] == name


def test_most_blocked_third_wins():
    depth = _far()
    depth[30:, 0:30] = 0.8  # left: fully blocked
    depth[50:, 30:60] = 0.8  # center: partly blocked
    assert OBSTACLE_CLASSES[obstacle_label(depth)] == "left"


def test_near_things_overhead_are_ignored():
    depth = _far()
    depth[:15, :] = 0.5  # ceiling-level clutter in the top quarter
    assert OBSTACLE_CLASSES[obstacle_label(depth)] == "none"


def test_missing_depth_is_not_an_obstacle():
    depth = _far()
    depth[30:, :] = 0.0  # Kinect holes are stored as zeros
    assert OBSTACLE_CLASSES[obstacle_label(depth)] == "none"


# --- COCO crop box --------------------------------------------------------------------


def test_crop_box_is_square_and_contains_the_object():
    x0, y0, side = square_crop_box((100, 50, 40, 80), img_w=640, img_h=480, pad=0.1)
    assert x0 <= 100 and y0 <= 50
    assert x0 + side >= 140 and y0 + side >= 130


def test_crop_box_stays_inside_the_image():
    x0, y0, side = square_crop_box((600, 440, 40, 40), img_w=640, img_h=480, pad=0.5)
    assert x0 >= 0 and y0 >= 0
    assert x0 + side <= 640 and y0 + side <= 480


# --- loader ---------------------------------------------------------------------------


def _write_processed(root, name, images, labels, split):
    (root / name).mkdir(parents=True)
    np.savez(root / name / "processed.npz", images=images, labels=labels, split=split)


def test_loader_returns_float_rgb_and_respects_split(tmp_path):
    images = np.arange(6 * 4 * 5, dtype=np.uint8).reshape(6, 4, 5)  # grayscale
    _write_processed(tmp_path, "toy", images, np.arange(6), np.array([0, 0, 0, 0, 1, 1]))
    x, y = load_dataset("toy", root=tmp_path, split="test")
    assert x.shape == (2, 3, 4, 5) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert y.tolist() == [4, 5]
    torch.testing.assert_close(x[:, 0], x[:, 2])  # gray copied to all channels


def test_loader_keeps_color_and_limits(tmp_path):
    images = np.random.default_rng(0).integers(0, 255, (5, 4, 5, 3), dtype=np.uint8)
    _write_processed(tmp_path, "toy", images, np.arange(5), np.zeros(5, dtype=int))
    x, y = load_dataset("toy", root=tmp_path, split="train", limit=3)
    assert x.shape == (3, 3, 4, 5) and len(y) == 3
    torch.testing.assert_close(x[0, 1], torch.from_numpy(images[0, :, :, 1]).float() / 255)


def test_missing_dataset_points_to_the_download_script(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_dataset("emnist", root=tmp_path)
