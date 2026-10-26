"""Phase 6: render the pilot's percepts once, so the volunteer page needs no data or networks.

For each task, test images are chosen so every encoder gets the same classes but different
pictures (``choose_images``), plus a few practice trials. Each image is encoded exactly as in
the benchmark (same patient, same encoding resolution), then rendered at a higher resolution
for human eyes. Writes PNGs and ``manifest.json`` to the stimuli folder.

Usage:
    python scripts/make_pilot_stimuli.py [--config configs/phase6_pilot.yaml] [--quick]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from phosphene_lab.config import load_config
from phosphene_lab.demo.pilot import Trial, choose_images
from phosphene_lab.eval.benchmark import build_encoders, encode, make_patient
from phosphene_lab.eval.tasks import TASKS, load_task
from phosphene_lab.simulator.phosphene import PhospheneRenderer, RenderConfig


def save_gray(field: torch.Tensor, path: Path, px: int) -> None:
    img = Image.fromarray((field.clamp(0, 1).numpy() * 255).astype(np.uint8))
    img.resize((px, px), Image.BILINEAR).save(path)


def save_rgb(image: torch.Tensor, path: Path, px: int) -> None:
    arr = (image.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr)
    scale = px / max(img.size)  # enlarge small images (28-px letters) as well as shrink
    img.resize((round(img.width * scale), round(img.height * scale)), Image.BILINEAR).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6_pilot.yaml")
    parser.add_argument("--quick", action="store_true", help="a handful of trials")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    rng = np.random.default_rng(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    out = Path(cfg["stimuli_dir"])
    if out.exists():
        shutil.rmtree(out)
    patient = make_patient(cfg["n_electrodes"], cfg["dropout"], cfg["variability"],
                           cfg["patient_seed"])  # fmt: skip
    pmap = patient.true_phosphene_map()
    encode_render = RenderConfig(**cfg["encode_render"])
    display = PhospheneRenderer(
        pmap, dataclasses.replace(encode_render, resolution=cfg["display_resolution"])
    )
    px = cfg["display_px"]

    trials: list[Trial] = []
    classes: dict[str, list[str]] = {}
    for name, tcfg in cfg["tasks"].items():
        task = TASKS[name]
        classes[name] = list(task.classes)
        images, labels, targets = load_task(task, "test", cfg["data_root"], cfg["limits"][name])
        labels_np = labels.numpy()
        encoders = build_encoders(cfg["encoders"], task.mode, name, encode_render,
                                  cfg["checkpoint_dir"])  # fmt: skip
        bad = set(cfg.get("exclude_images", {}).get(name, []))  # mislabeled, found by eye
        chosen_classes = rng.choice(task.n_classes, size=tcfg["classes"], replace=False)
        scored = choose_images(labels_np, cfg["encoders"], chosen_classes, tcfg["per_class"],
                               rng, exclude=bad)  # fmt: skip
        used = bad | {i for idx in scored.values() for i in idx}
        # Practice: one picture per encoder, cycling through the encoders, never a scored one.
        practice_classes = rng.choice(task.n_classes, size=tcfg["practice"], replace=True)
        practice: dict[str, list[int]] = {e: [] for e in cfg["encoders"]}
        for k, c in enumerate(practice_classes):
            enc = cfg["encoders"][k % len(cfg["encoders"])]
            pick = choose_images(labels_np, [enc], [c], 1, rng, exclude=used)[enc]
            practice[enc] += pick
            used.update(pick)

        folder = out / name
        folder.mkdir(parents=True)
        for is_practice, picks in ((True, practice), (False, scored)):
            for enc, idx in picks.items():
                if not idx:
                    continue
                stim = encode(encoders[enc], images[idx], targets[idx], pmap)
                with torch.no_grad():
                    percepts = display.render(stim)
                for k, i in enumerate(idx):
                    stem = f"{'practice' if is_practice else 'trial'}_{enc}_{i}"
                    save_gray(percepts[k], folder / f"{stem}.png", px)
                    save_rgb(images[i], folder / f"{stem}_original.png", px)
                    trials.append(Trial(
                        name, enc, task.classes[int(labels_np[i])], f"{name}/{stem}.png",
                        f"{name}/{stem}_original.png", practice=is_practice,
                    ))  # fmt: skip
        n_scored = sum(len(v) for v in scored.values())
        print(f"{name}: {n_scored} scored + {tcfg['practice']} practice trials")

    manifest = {
        "classes": classes,
        "trials": [dataclasses.asdict(t) for t in trials],
        "condition": {
            k: cfg[k]
            for k in ("n_electrodes", "dropout", "variability", "patient_seed", "encoders")
        },  # fmt: skip
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {len(trials)} trials to {out}")


if __name__ == "__main__":
    main()
