"""Task datasets: loaders for preprocessed files plus the labelling rules used to build them.

``scripts/download_data.py`` downloads each raw dataset once and writes
``data/<name>/processed.npz`` with ``images`` (uint8, (N, H, W) or (N, H, W, 3)), ``labels``
(int) and ``split`` (0 = train, 1 = test). Everything downstream reads only these files.

* ``emnist``: EMNIST letters, 26 classes (reading).
* ``coco``: a 10-class COCO val2017 subset, each image cropped to its dominant object.
* ``nyu``: NYU Depth V2 labeled frames with an obstacle label derived from depth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# The ten categories that are most often the largest object in a val2017 image (>= 70 each
# at 5% of the frame), so classes can be balanced without the 448 MB train2017 annotations.
COCO_CLASSES = (
    "person", "dining table", "train", "bus", "bed",
    "truck", "couch", "toilet", "motorcycle", "cat",
)  # fmt: skip
OBSTACLE_CLASSES = ("none", "left", "center", "right")
SPLITS = {"train": 0, "test": 1}


def obstacle_label(
    depth_m: np.ndarray, near_m: float = 1.5, min_fraction: float = 0.15, top_skip: float = 1 / 3
) -> int:
    """Index into OBSTACLE_CLASSES for one depth map (meters, 0 = missing).

    The top ``top_skip`` of the frame (ceiling, wall tops) is ignored. The rest is split into
    left/center/right thirds; a third is blocked when at least ``min_fraction`` of its valid
    pixels are nearer than ``near_m``. The most blocked third is the label.
    """
    ground = depth_m[int(depth_m.shape[0] * top_skip) :]
    fractions = []
    for third in np.array_split(ground, 3, axis=1):
        valid = third > 0
        near = valid & (third < near_m)
        fractions.append(near.sum() / max(valid.sum(), 1))
    best = int(np.argmax(fractions))
    return best + 1 if fractions[best] >= min_fraction else 0


def square_crop_box(
    bbox_xywh: tuple[float, float, float, float], img_w: int, img_h: int, pad: float = 0.1
) -> tuple[int, int, int]:
    """(x0, y0, side) of a square around a box, padded by ``pad`` and shifted inside the image."""
    x, y, w, h = bbox_xywh
    side = int(round(min(max(w, h) * (1 + 2 * pad), img_w, img_h)))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(round(min(max(cx - side / 2, 0), img_w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), img_h - side)))
    return x0, y0, side


def load_dataset(
    name: str, root: str | Path = "data", split: str = "train", limit: int | None = None
) -> tuple[Tensor, Tensor]:
    """Images as float RGB in [0, 1], shape (N, 3, H, W), and integer labels (N,)."""
    path = Path(root) / name / "processed.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python scripts/download_data.py --dataset {name}` first"
        )
    with np.load(path) as f:
        keep = np.flatnonzero(f["split"] == SPLITS[split])[:limit]
        images, labels = f["images"][keep], f["labels"][keep]
    x = torch.from_numpy(images).float() / 255
    x = x.unsqueeze(1).expand(-1, 3, -1, -1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
    return x.contiguous(), torch.from_numpy(labels).long()
