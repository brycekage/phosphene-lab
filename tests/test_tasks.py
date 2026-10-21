"""Tests for the benchmark task registry and loader."""

import numpy as np

from phosphene_lab.eval.tasks import TASKS, load_task


def test_registry_matches_the_spec():
    assert {t: TASKS[t].n_classes for t in TASKS} == {"letters": 26, "objects": 10, "obstacles": 4}
    assert {TASKS[t].dataset for t in TASKS} == {"emnist", "coco", "nyu"}


def test_load_task_returns_images_labels_and_cached_targets(tmp_path):
    (tmp_path / "emnist").mkdir()
    images = np.zeros((4, 8, 8), dtype=np.uint8)
    images[:, 2:6, 3:5] = 255  # a white stroke
    np.savez(
        tmp_path / "emnist" / "processed.npz",
        images=images,
        labels=np.arange(4),
        split=np.zeros(4, dtype=int),
    )
    x, y, targets = load_task(TASKS["letters"], "train", root=tmp_path)
    assert x.shape == (4, 3, 8, 8) and y.tolist() == [0, 1, 2, 3]
    assert targets.shape == (4, 1, 8, 8)
    assert targets[0, 0, 3, 3] == 1 and targets[0, 0, 0, 0] == 0
    assert len(list((tmp_path / "emnist").glob("targets_reading_train_*.npy"))) == 1


def test_target_cache_follows_the_images(tmp_path):
    (tmp_path / "emnist").mkdir()
    path = tmp_path / "emnist" / "processed.npz"
    images = np.zeros((2, 8, 8), dtype=np.uint8)
    images[0, 2:6, 3:5] = 255
    np.savez(path, images=images, labels=np.arange(2), split=np.zeros(2, dtype=int))
    _, _, first = load_task(TASKS["letters"], "train", root=tmp_path)
    np.savez(path, images=images[::-1].copy(), labels=np.arange(2), split=np.zeros(2, dtype=int))
    _, _, second = load_task(TASKS["letters"], "train", root=tmp_path)
    assert first[0].sum() > 0 and second[0].sum() == 0  # reordered data, fresh targets
