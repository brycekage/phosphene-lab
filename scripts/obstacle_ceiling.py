"""How much obstacle information each encoder input carries before the simulator.

Trains the benchmark's observer directly on (a) the photo shrunk to E3's 48-px input, in
grayscale, and (b) the E2 navigation scene at the same size, with no phosphenes in between.
These are ceilings for an encoder that reads that input: E3 trained on the photo alone cannot
be expected to beat (a).

Usage:
    python scripts/obstacle_ceiling.py --config configs/obstacle_ceiling.yaml [--quick]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from phosphene_lab.config import load_config
from phosphene_lab.eval.benchmark import load_task_data
from phosphene_lab.eval.observer import accuracy, train_observer


def shrink(images: torch.Tensor, side: int) -> torch.Tensor:
    h, w = images.shape[-2:]
    scale = side / max(h, w)
    return F.interpolate(images, size=(round(h * scale), round(w * scale)), mode="area")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/obstacle_ceiling.yaml")
    parser.add_argument("--quick", action="store_true", help="tiny data, one seed")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    data = load_task_data("obstacles", cfg["data_root"], cfg["limits"])
    side = cfg["side_px"]
    inputs = {
        "photo_gray": (
            shrink(data.train_images.mean(1, keepdim=True), side),
            shrink(data.test_images.mean(1, keepdim=True), side),
        ),
        "e2_scene": (shrink(data.train_targets, side), shrink(data.test_targets, side)),
    }
    rows = []
    for name, (train, test) in inputs.items():
        for seed in cfg["seeds"]:
            observer = train_observer(
                train, data.train_labels, data.n_classes, epochs=cfg["observer_epochs"],
                seed=seed, width=cfg["observer_width"],
            )  # fmt: skip
            acc = accuracy(observer, test, data.test_labels)
            rows.append({"input": name, "side_px": side, "seed": seed, "accuracy": round(acc, 4)})
            print(f"{name} seed {seed}: {acc:.3f}")

    out_csv = Path(cfg["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
