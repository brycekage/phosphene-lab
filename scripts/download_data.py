"""Phase 2: download the task datasets and write data/<name>/processed.npz for each.

Raw downloads are cached in data/raw (re-runs skip them). Also writes a per-class count table
so the README can report exactly what the benchmark was trained and tested on.

* emnist: EMNIST letters via torchvision (~560 MB archive of every EMNIST split).
* coco:   10 classes from COCO val2017; each image is cropped to its largest object.
* nyu:    NYU Depth V2 labeled (~3 GB .mat); obstacle labels come from the depth map and the
          train/test split is by scene.

Usage:
    python scripts/download_data.py --config configs/data.yaml [--dataset all] [--quick]
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from phosphene_lab.config import load_config
from phosphene_lab.datasets import (
    COCO_CLASSES,
    OBSTACLE_CLASSES,
    obstacle_label,
    square_crop_box,
)

COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
NYU_LABELED = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    return dest


def save(out: Path, images: np.ndarray, labels: np.ndarray, split: np.ndarray, **extra) -> None:
    # Shuffle rows so any "first N images" subset covers every class (seeded, so stable).
    order = np.random.default_rng(0).permutation(len(labels))
    extra = {k: v[order] for k, v in extra.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, images=images[order], labels=labels[order], split=split[order], **extra
    )
    print(f"wrote {out}: {len(labels)} images, {int((split == 1).sum())} test")


def build_emnist(cfg: dict, raw: Path, out: Path, seed: int):
    from torchvision.datasets import EMNIST

    rng = np.random.default_rng(seed)
    parts = []
    for train, n in ((True, cfg["n_train"]), (False, cfg["n_test"])):
        ds = EMNIST(str(raw / "emnist"), split="letters", train=train, download=True)
        # The test split is sorted by letter, so shuffle before taking a subset.
        keep = rng.permutation(len(ds.targets))[:n]
        # EMNIST stores each image transposed relative to how it is written.
        parts.append((ds.data[keep].transpose(1, 2).numpy(), ds.targets[keep].numpy() - 1))
    images = np.concatenate([p[0] for p in parts])
    labels = np.concatenate([p[1] for p in parts])
    split = np.repeat([0, 1], [len(parts[0][1]), len(parts[1][1])])
    save(out, images, labels, split)
    return labels, split, [chr(ord("a") + i) for i in range(26)]


def build_coco(cfg: dict, raw: Path, out: Path, seed: int):
    ann_path = raw / "instances_val2017.json"
    if not ann_path.exists():
        with zipfile.ZipFile(fetch(COCO_ANNOTATIONS, raw / "annotations_trainval2017.zip")) as z:
            ann_path.write_bytes(z.read("annotations/instances_val2017.json"))
    coco = json.loads(ann_path.read_text())
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    largest: dict[int, dict] = {}
    for ann in coco["annotations"]:
        if not ann["iscrowd"] and ann["area"] > largest.get(ann["image_id"], {}).get("area", 0):
            largest[ann["image_id"]] = ann
    chosen: dict[str, list[dict]] = {name: [] for name in COCO_CLASSES}
    for image_id in sorted(largest):
        ann, im = largest[image_id], images[image_id]
        name = cat_name[ann["category_id"]]
        big_enough = ann["area"] >= cfg["min_area_fraction"] * im["width"] * im["height"]
        if name in chosen and big_enough and len(chosen[name]) < cfg["per_class"]:
            chosen[name].append(ann)

    picks = [(ann, c) for c, name in enumerate(COCO_CLASSES) for ann in chosen[name]]
    folder = raw / "coco_val2017"

    def crop(item):
        ann, _ = item
        im = images[ann["image_id"]]
        path = fetch(im["coco_url"], folder / im["file_name"])
        bgr = cv2.imread(str(path))
        x0, y0, side = square_crop_box(ann["bbox"], im["width"], im["height"], cfg["crop_pad"])
        patch = bgr[y0 : y0 + side, x0 : x0 + side]
        patch = cv2.resize(patch, (cfg["size"], cfg["size"]), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(16) as pool:
        crops = list(pool.map(crop, picks))
    labels = np.array([c for _, c in picks])
    split = np.zeros(len(labels), dtype=int)
    rng = np.random.default_rng(seed)
    for c in range(len(COCO_CLASSES)):
        idx = rng.permutation(np.flatnonzero(labels == c))
        split[idx[: int(round(len(idx) * cfg["test_fraction"]))]] = 1
    save(out, np.stack(crops), labels, split)
    return labels, split, list(COCO_CLASSES)


def build_nyu(cfg: dict, raw: Path, out: Path, seed: int):
    import h5py

    with h5py.File(fetch(NYU_LABELED, raw / "nyu_depth_v2_labeled.mat"), "r") as f:
        n = len(f["images"]) if cfg.get("max_frames") is None else cfg["max_frames"]
        scenes = ["".join(map(chr, f[ref][()].flatten())) for ref in f["scenes"][0][:n]]
        rgbs, depths, labels = [], [], []
        size = (cfg["width"], cfg["height"])
        for i in range(n):
            # MATLAB v7.3 stores arrays column-major: (C, W, H) -> (H, W, C).
            rgb = np.ascontiguousarray(f["images"][i].transpose(2, 1, 0))
            depth = np.ascontiguousarray(f["depths"][i].T.astype(np.float32))
            labels.append(obstacle_label(depth, cfg["near_m"], cfg["min_fraction"]))
            rgbs.append(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
            depths.append(cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST))
    unique = sorted(set(scenes))
    rng = np.random.default_rng(seed)
    test_scenes = set(rng.permutation(unique)[: int(round(len(unique) * cfg["test_fraction"]))])
    split = np.array([int(s in test_scenes) for s in scenes])
    labels = np.array(labels)
    save(out, np.stack(rgbs), labels, split, depth=np.stack(depths).astype(np.float16))
    return labels, split, list(OBSTACLE_CLASSES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--dataset", default="all", choices=["all", "emnist", "coco", "nyu"])
    parser.add_argument("--quick", action="store_true", help="small subsets for smoke tests")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    root, raw = Path(cfg["root"]), Path(cfg["raw"])

    names = ["emnist", "coco", "nyu"] if args.dataset == "all" else [args.dataset]
    rows = []
    for name in names:
        out = root / name / "processed.npz"
        if name == "emnist":
            labels, split, classes = build_emnist(cfg[name], raw, out, cfg["seed"])
        elif name == "coco":
            labels, split, classes = build_coco(cfg[name], raw, out, cfg["seed"])
        else:
            labels, split, classes = build_nyu(cfg[name], raw, out, cfg["seed"])
        for s, s_name in ((0, "train"), (1, "test")):
            counts = Counter(labels[split == s].tolist())
            rows += [[name, s_name, cls, counts.get(c, 0)] for c, cls in enumerate(classes)]

    summary = Path(cfg["summary_csv"])
    summary.parent.mkdir(parents=True, exist_ok=True)
    if summary.exists():  # keep rows for datasets not rebuilt in this run
        with summary.open(newline="") as fh:
            rows = [r for r in list(csv.reader(fh))[1:] if r[0] not in names] + rows
    with summary.open("w", newline="") as fh:
        csv.writer(fh).writerows([["dataset", "split", "class", "count"], *rows])
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
