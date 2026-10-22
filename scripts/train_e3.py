"""Phase 2: train the learned encoder E3 for each task, through the differentiable simulator.

Writes one checkpoint per task (encoder + its jointly trained observer), a CSV with wall-clock
time and final losses (to check the < 1 h compute budget), and a loss-curve figure.

Usage:
    python scripts/train_e3.py --config configs/phase2_e3.yaml [--quick]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from phosphene_lab.config import load_config, with_overrides  # noqa: E402
from phosphene_lab.encoders.learned import E3TrainConfig, train_learned_encoder  # noqa: E402
from phosphene_lab.eval.tasks import TASKS, load_task  # noqa: E402
from phosphene_lab.simulator.phosphene import RenderConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase2_e3.yaml")
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke tests")
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), help="override config")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    if args.tasks:
        cfg["tasks"] = args.tasks
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg.get("num_threads", torch.get_num_threads()))

    base = dict(cfg["train"])
    base["electrode_counts"] = tuple(base["electrode_counts"])
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in cfg["tasks"]:
        task = TASKS[name]
        train = with_overrides(base, cfg.get("task_overrides"), name)
        train_cfg = E3TrainConfig(**train, render=RenderConfig(**cfg["render"]), seed=cfg["seed"])
        images, labels, targets = load_task(task, "train", cfg["data_root"], cfg["train_limit"])
        start = time.perf_counter()
        encoder, observer, history = train_learned_encoder(
            images, targets, labels, task.n_classes, train_cfg
        )
        seconds = time.perf_counter() - start
        torch.save(
            {"encoder": encoder.state_dict(), "observer": observer.state_dict(),
             "width": train_cfg.width, "max_side": train_cfg.max_side,
             "n_classes": task.n_classes, "history": history},
            ckpt_dir / f"e3_{name}.pt",
        )  # fmt: skip
        tail = history[-max(1, len(history) // 10) :]  # last 10% of steps
        rows.append(
            [name, len(labels), train_cfg.steps, round(seconds, 1),
             round(seconds / train_cfg.steps, 3)]
            + [round(float(np.mean([h[k] for h in tail])), 4) for k in ("recon", "task", "power")]
            + [train_cfg.power_weight]
        )  # fmt: skip
        print(f"{name}: {seconds:.0f} s, final recon {rows[-1][5]}, task {rows[-1][6]}")

    # Tasks can be trained in separate runs (--tasks): keep other tasks' rows and curves.
    out_csv = Path(cfg["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        with out_csv.open(newline="") as fh:
            kept = [r for r in list(csv.reader(fh))[1:] if r[0] not in cfg["tasks"]]
        # Rows written before the power_weight column existed used the base setting.
        kept = [r if len(r) == 9 else [*r, base["power_weight"]] for r in kept]
        rows = kept + rows
    curves = {
        name: torch.load(ckpt_dir / f"e3_{name}.pt", weights_only=True)["history"]
        for name in TASKS
        if (ckpt_dir / f"e3_{name}.pt").exists()
        and "history" in torch.load(ckpt_dir / f"e3_{name}.pt", weights_only=True)
    }
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["task", "n_train", "steps", "seconds", "seconds_per_step",
             "final_recon", "final_task_ce", "final_power", "power_weight"]
        )  # fmt: skip
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for name, history in curves.items():
        steps = [h["step"] for h in history]
        for ax, key in zip(axes, ("recon", "task", "power"), strict=True):
            values = np.array([h[key] for h in history])
            window = max(1, len(values) // 30)
            smooth = np.convolve(values, np.ones(window) / window, mode="valid")
            ax.plot(steps[: len(smooth)], smooth, label=name)
            ax.set_title({"recon": "reconstruction MSE", "task": "observer cross-entropy",
                          "power": "mean current / max"}[key])  # fmt: skip
            ax.set_xlabel("step")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(cfg["output_png"], dpi=120)
    print(f"wrote {out_csv} and {cfg['output_png']}")


if __name__ == "__main__":
    main()
