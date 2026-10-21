"""The three benchmark tasks and how to load them.

Each task pairs a dataset with the E2 mode whose simplified scene suits it:

* letters:   EMNIST letters, 26 classes, "reading" mode.
* objects:   10-class COCO subset, "navigation" mode (the cropped object is the near thing).
* obstacles: NYU Depth V2, none/left/center/right, "navigation" mode.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from torch import Tensor

from phosphene_lab.datasets import COCO_CLASSES, OBSTACLE_CLASSES, load_dataset
from phosphene_lab.encoders.scene import SceneEncoder, SceneModels, cached_targets


@dataclass(frozen=True)
class Task:
    name: str
    dataset: str
    mode: str
    classes: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


TASKS = {
    "letters": Task("letters", "emnist", "reading", tuple(chr(ord("a") + i) for i in range(26))),
    "objects": Task("objects", "coco", "navigation", COCO_CLASSES),
    "obstacles": Task("obstacles", "nyu", "navigation", OBSTACLE_CLASSES),
}


def load_task(
    task: Task,
    split: str,
    root: str | Path = "data",
    limit: int | None = None,
    scene_models: SceneModels | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Images (N, 3, H, W), labels (N,), and cached E2 targets (N, 1, H, W)."""
    images, labels = load_dataset(task.dataset, root, split, limit)
    encoder = SceneEncoder(task.mode, models=scene_models)
    # Key the cache by content, so a rebuilt or reordered dataset never reuses old targets.
    digest = hashlib.md5(images.numpy().tobytes()).hexdigest()[:12]
    cache = Path(root) / task.dataset / f"targets_{task.mode}_{split}_{digest}.npy"
    return images, labels, cached_targets(encoder, images, cache)
