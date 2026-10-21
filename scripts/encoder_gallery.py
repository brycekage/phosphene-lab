"""Phase 2: side-by-side figure of what each encoder lets a patient see.

For a few test images per task, renders: the original (placed where the implant can draw),
the E2 simplified target, and the percepts from E0 naive, E1 edges, E2 scene, and E3 learned,
at each electrode count. The patient is a held-out virtual patient. Also writes a CSV with
active electrodes and mean current per percept.

Usage:
    python scripts/encoder_gallery.py --config configs/phase2_gallery.yaml [--quick]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from phosphene_lab.config import load_config  # noqa: E402
from phosphene_lab.encoders.base import image_to_field  # noqa: E402
from phosphene_lab.encoders.edges import EdgeEncoder  # noqa: E402
from phosphene_lab.encoders.learned import load_e3  # noqa: E402
from phosphene_lab.encoders.naive import NaiveEncoder  # noqa: E402
from phosphene_lab.encoders.scene import PretrainedSceneModels, SceneEncoder  # noqa: E402
from phosphene_lab.eval.tasks import TASKS, load_task  # noqa: E402
from phosphene_lab.simulator.array import ElectrodeArray  # noqa: E402
from phosphene_lab.simulator.patient import VirtualPatient  # noqa: E402
from phosphene_lab.simulator.phosphene import PhospheneRenderer, RenderConfig  # noqa: E402

ENCODERS = ("E0 naive", "E1 edges", "E2 scene", "E3 learned")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_gallery.yaml")
    parser.add_argument("--quick", action="store_true", help="one image, one array size")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    torch.manual_seed(cfg["seed"])
    render = RenderConfig(**cfg["render"])
    scene_models = PretrainedSceneModels()

    panels, rows = [], []  # panels: (row title, [(column title, image)])
    for name in cfg["tasks"]:
        task = TASKS[name]
        images, labels, targets = load_task(task, "test", cfg["data_root"], None, scene_models)
        picks = torch.linspace(0, len(labels) - 1, cfg["images_per_task"]).long()
        e3, _ = load_e3(str(Path(cfg["checkpoint_dir"]) / f"e3_{name}.pt"), render)
        encoders = dict(
            zip(
                ENCODERS,
                (NaiveEncoder(config=render), EdgeEncoder(config=render),
                 SceneEncoder(task.mode, scene_models, config=render), e3),
                strict=True,
            )
        )  # fmt: skip
        for n in cfg["electrode_counts"]:
            patient = VirtualPatient(ElectrodeArray.grid(n), seed=cfg["patient_seed"])
            pmap = patient.true_phosphene_map()
            renderer = PhospheneRenderer(pmap, render)
            for i in picks.tolist():
                image, target = images[i : i + 1], targets[i : i + 1]
                columns = [
                    ("original", image_to_field(image, pmap, render)[0].permute(1, 2, 0)),
                    ("E2 target", image_to_field(target, pmap, render)[0, 0]),
                ]
                with torch.no_grad():
                    for enc_name, encoder in encoders.items():
                        stim = encoder(image, pmap)
                        columns.append((enc_name, renderer.render(stim)[0]))
                        active = stim[0] > 0
                        mean_current = stim[0][active].mean().item() if active.any() else 0.0
                        rows.append(
                            [name, i, task.classes[int(labels[i])], n, enc_name,
                             int(active.sum()), round(mean_current, 1)]
                        )  # fmt: skip
                panels.append((f"{name} '{task.classes[int(labels[i])]}'\n{n} electrodes", columns))
        print(f"{name}: done")

    n_cols = len(panels[0][1])
    fig, axes = plt.subplots(len(panels), n_cols, figsize=(1.9 * n_cols, 1.9 * len(panels)))
    axes = axes.reshape(len(panels), n_cols)
    for r, (title, columns) in enumerate(panels):
        for c, (col_title, img) in enumerate(columns):
            ax = axes[r, c]
            ax.imshow(img.clamp(0, 1).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_title, fontsize=9)
        axes[r, 0].set_ylabel(title, fontsize=8)
    fig.tight_layout()
    out_png = Path(cfg["output_png"])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)

    with Path(cfg["output_csv"]).open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["task", "image", "label", "n_electrodes", "encoder", "active", "mean_current_uA"]
        )
        writer.writerows(rows)
    print(f"wrote {out_png} and {cfg['output_csv']}")


if __name__ == "__main__":
    main()
