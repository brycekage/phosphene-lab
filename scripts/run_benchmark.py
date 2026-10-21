"""Phase 3: benchmark every encoder on every task across array sizes, dropout, and patients.

For each cell (task, encoder, electrode count, dropout, variability, seed): build a held-out
virtual patient, encode the task's train and test images, render percepts with the patient's
true map, train a fresh observer on this encoder's train percepts, and score it on the test
percepts. Within a seed every encoder sees the same patient, so comparisons are paired.

Writes every cell to a CSV as it goes, then a mean +/- std summary and three figures:
accuracy vs electrode count (headline), accuracy vs dropout, accuracy vs variability.

Usage:
    python scripts/run_benchmark.py --config configs/benchmark.yaml [--quick]
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

from phosphene_lab.config import load_config  # noqa: E402
from phosphene_lab.datasets import load_dataset  # noqa: E402
from phosphene_lab.eval.benchmark import (  # noqa: E402
    build_encoders,
    evaluate_cell,
    load_task_data,
    make_patient,
)
from phosphene_lab.eval.tasks import TASKS  # noqa: E402
from phosphene_lab.simulator.phosphene import RenderConfig  # noqa: E402

FIELDS = ["task", "encoder", "n_electrodes", "dropout", "variability", "seed",
          "accuracy", "ssim", "active_electrodes", "total_current_uA", "seconds"]  # fmt: skip
METRICS = ("accuracy", "ssim", "active_electrodes", "total_current_uA")
KEYS = ("task", "encoder", "n_electrodes", "dropout", "variability")


def cells(cfg: dict) -> list[tuple[int, float, str]]:
    main, var = cfg["main_sweep"], cfg["variability_sweep"]
    grid = [(n, d, main["variability"]) for n in main["electrode_counts"] for d in main["dropouts"]]
    grid += [(var["electrode_count"], var["dropout"], level) for level in var["levels"]]
    return list(dict.fromkeys(grid))  # drop duplicates, keep order


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in KEYS), []).append(row)
    out = []
    for key, members in groups.items():
        entry = dict(zip(KEYS, key, strict=True)) | {"n_seeds": len(members)}
        for m in METRICS:
            values = np.array([float(r[m]) for r in members])
            entry[f"{m}_mean"] = round(float(values.mean()), 4)
            entry[f"{m}_std"] = round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4)
        out.append(entry)
    return out


def plot(summary: list[dict], cfg: dict) -> None:
    present = {s["task"] for s in summary}  # a partial sweep may not cover every task yet
    tasks = [t for t in cfg["tasks"] if t in present]
    encoders = cfg["encoders"]
    main, var = cfg["main_sweep"], cfg["variability_sweep"]

    def series(task, encoder, x_key, **fixed):
        pts = [s for s in summary if s["task"] == task and s["encoder"] == encoder
               and all(s[k] == v for k, v in fixed.items())]  # fmt: skip
        pts.sort(key=lambda s: s[x_key])
        return ([s[x_key] for s in pts], [s["accuracy_mean"] for s in pts],
                [s["accuracy_std"] for s in pts])  # fmt: skip

    figures = [
        ("headline_png", "n_electrodes", "electrodes",
         {"dropout": 0.0, "variability": main["variability"]}),
        ("dropout_png", "dropout", "electrode dropout",
         {"n_electrodes": var["electrode_count"], "variability": main["variability"]}),
        ("variability_png", "variability", "patient variability",
         {"n_electrodes": var["electrode_count"], "dropout": var["dropout"]}),
    ]  # fmt: skip
    for out_key, x_key, x_label, fixed in figures:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.4), squeeze=False)
        for ax, task in zip(axes[0], tasks, strict=True):
            for encoder in encoders:
                x, y, err = series(task, encoder, x_key, **fixed)
                if x_key == "variability":  # categorical axis in the configured order
                    order = {lvl: i for i, lvl in enumerate(var["levels"])}
                    x = [order[v] for v in x]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=encoder)
            # Always answering the most common test class; for obstacles ("none" is 35% of
            # test frames) this is well above 1 / n_classes.
            ax.axhline(majority_baseline(task, cfg), color="gray", lw=0.8, ls="--")
            ax.set_title(task)
            ax.set_xlabel(x_label)
            ax.set_ylim(0, 1)
            if x_key == "n_electrodes":
                ax.set_xscale("log")
                ax.set_xticks(main["electrode_counts"], [str(n) for n in main["electrode_counts"]])
            if x_key == "variability":
                ax.set_xticks(range(len(var["levels"])), var["levels"])
        axes[0, 0].set_ylabel("observer accuracy\n(dashed: majority-class guess)")
        axes[0, -1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(cfg[out_key], dpi=120)
        plt.close(fig)


def majority_baseline(task_name: str, cfg: dict) -> float:
    """Accuracy of always guessing the most common class in the benchmark's test split."""
    task = TASKS[task_name]
    _, labels = load_dataset(
        task.dataset, cfg["data_root"], "test", cfg["limits"][task_name]["test"]
    )
    return labels.bincount().max().item() / len(labels)


def run_part(cfg, task_name, seed, data, encoders, render, path: Path) -> None:
    """Every grid cell for one task and seed. Each encoder sees the same patient."""
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for n, dropout, variability in cells(cfg):
            patient = make_patient(n, dropout, variability, cfg["patient_seed_offset"] + seed)
            for enc_name, encoder in encoders.items():
                t0 = time.perf_counter()
                result = evaluate_cell(
                    data, encoder, patient, render, cfg["observer_epochs"], seed,
                    observer_width=cfg["observer_width"],
                )  # fmt: skip
                row = {"task": task_name, "encoder": enc_name, "n_electrodes": n,
                       "dropout": dropout, "variability": variability, "seed": seed}  # fmt: skip
                row |= {k: round(v, 4) for k, v in result.items()}
                row["seconds"] = round(time.perf_counter() - t0, 2)
                writer.writerow(row)
                fh.flush()


def combine_parts(cfg: dict, parts_dir: Path, out_csv: Path) -> list[dict]:
    """Merge the configured tasks' part files into one CSV, with numeric types restored."""
    rows = []
    for name in cfg["tasks"]:
        for part in sorted(parts_dir.glob(f"{name}_seed*.csv")):
            with part.open(newline="") as fh:
                for r in csv.DictReader(fh):
                    r["n_electrodes"], r["seed"] = int(r["n_electrodes"]), int(r["seed"])
                    for k in ("dropout", *METRICS, "seconds"):
                        r[k] = float(r[k])
                    rows.append(r)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--quick", action="store_true", help="tiny sweep for smoke tests")
    # A long sweep can be split into several runs: each (task, seed) part is its own CSV in
    # <output>_parts/, and every run rebuilds the combined CSV, summary and figures.
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), help="run only these")
    parser.add_argument("--seeds", nargs="+", type=int, help="run only these seeds")
    parser.add_argument("--combine-only", action="store_true", help="only rebuild outputs")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    torch.manual_seed(cfg["seed"])
    render = RenderConfig(**cfg["render"])

    out_csv = Path(cfg["output_csv"])
    parts_dir = out_csv.parent / f"{out_csv.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    tasks = [] if args.combine_only else (args.tasks or cfg["tasks"])
    seeds = args.seeds if args.seeds is not None else cfg["seeds"]
    start = time.perf_counter()
    for name in tasks:
        data = load_task_data(name, cfg["data_root"], cfg["limits"][name])
        encoders = build_encoders(
            cfg["encoders"], TASKS[name].mode, name, render, cfg["checkpoint_dir"]
        )
        for seed in seeds:
            run_part(cfg, name, seed, data, encoders, render, parts_dir / f"{name}_seed{seed}.csv")
            print(f"{name} seed {seed}: {(time.perf_counter() - start) / 60:.1f} min elapsed")

    rows = combine_parts(cfg, parts_dir, out_csv)
    summary = summarize(rows)
    with Path(cfg["summary_csv"]).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    plot(summary, cfg)
    print(f"wrote {out_csv}, {cfg['summary_csv']} and figures "
          f"in {(time.perf_counter() - start) / 60:.1f} min")  # fmt: skip


if __name__ == "__main__":
    main()
