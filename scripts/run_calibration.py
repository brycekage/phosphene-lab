"""Phase 4: queries-to-accuracy for exhaustive, random-order, and active calibration.

For each patient model and seed, runs every method on the same virtual patient up to the
query budget, snapshotting the estimated map every few queries. Writes the error curves
(CSV and figure, with clinic time at ~5 s per query) and, for each method, how many queries
it needs to match the location and threshold error that one full exhaustive pass reaches.

Usage:
    python scripts/run_calibration.py --config configs/phase4_calibration.yaml [--quick]
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

from phosphene_lab.calibration.active import ActiveCalibrator  # noqa: E402
from phosphene_lab.calibration.baselines import (  # noqa: E402
    RandomCalibrator,
    SequentialCalibrator,
)
from phosphene_lab.calibration.common import (  # noqa: E402
    SECONDS_PER_QUERY,
    CalibrationConfig,
    map_errors,
)
from phosphene_lab.config import load_config  # noqa: E402
from phosphene_lab.simulator.array import ElectrodeArray  # noqa: E402
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient  # noqa: E402

CALIBRATORS = {
    "exhaustive": SequentialCalibrator,
    "random": RandomCalibrator,
    "active": ActiveCalibrator,
}
ERRORS = ("location_error_deg", "threshold_error_uA")


def queries_to_reach(queries: np.ndarray, curve: np.ndarray, target: float) -> float:
    """First checkpoint where the (seed-averaged) error is at or below target; NaN if never."""
    hits = np.flatnonzero(curve <= target)
    return float(queries[hits[0]]) if len(hits) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4_calibration.yaml")
    parser.add_argument("--quick", action="store_true", help="small array, short budget")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    torch.manual_seed(cfg["seed"])

    array = ElectrodeArray.grid(cfg["n_electrodes"])
    checkpoints = list(range(0, cfg["max_queries"] + 1, cfg["checkpoint_every"]))
    rows = []
    start = time.perf_counter()
    for model_name, overrides in cfg["patient_models"].items():
        params = PatientParams(**overrides)
        calib_cfg = CalibrationConfig.from_population(params)
        for seed in cfg["seeds"]:
            for method in cfg["methods"]:
                patient = VirtualPatient(array, params, seed=cfg["patient_seed_offset"] + seed)
                snapshots = CALIBRATORS[method](array, calib_cfg, seed=seed).run(
                    patient, checkpoints
                )
                truth = patient.true_phosphene_map()  # evaluation only, after calibration
                for queries, estimate in snapshots:
                    errors = map_errors(estimate, truth)
                    rows.append(
                        {"patient_model": model_name, "method": method, "seed": seed,
                         "queries": queries,
                         "clinic_minutes": round(queries * SECONDS_PER_QUERY / 60, 2)}
                        | {k: round(v, 4) for k, v in errors.items()}
                    )  # fmt: skip
                print(f"{model_name} seed {seed} {method}: "
                      f"{time.perf_counter() - start:.0f} s elapsed")  # fmt: skip

    out_csv = Path(cfg["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def curve(model_name, method, key):
        per_seed = [
            [r[key] for r in rows if r["patient_model"] == model_name
             and r["method"] == method and r["seed"] == s]
            for s in cfg["seeds"]
        ]  # fmt: skip
        values = np.array(per_seed)
        return values.mean(axis=0), values.std(axis=0)

    q = np.array(checkpoints)
    targets = []
    models = list(cfg["patient_models"])
    fig, axes = plt.subplots(len(models), 2, figsize=(10, 3.6 * len(models)), squeeze=False)
    for r, model_name in enumerate(models):
        for c, key in enumerate(ERRORS):
            ax = axes[r, c]
            target = curve(model_name, "exhaustive", key)[0][-1]  # one full exhaustive pass
            for method in cfg["methods"]:
                mean, std = curve(model_name, method, key)
                ax.plot(q, mean, label=method)
                ax.fill_between(q, mean - std, mean + std, alpha=0.2)
                needed = queries_to_reach(q, mean, target)
                targets.append(
                    {"patient_model": model_name, "metric": key, "method": method,
                     "target": round(float(target), 4), "queries_needed": needed,
                     "clinic_minutes": round(needed * SECONDS_PER_QUERY / 60, 1)}
                )  # fmt: skip
            ax.axhline(target, color="gray", lw=0.8, ls="--")
            ax.set_title(f"{model_name} patients: {key.replace('_', ' ')}")
            ax.set_xlabel("queries")
            top = ax.secondary_xaxis(
                "top", functions=(lambda x: x * SECONDS_PER_QUERY / 60,
                                  lambda m: m * 60 / SECONDS_PER_QUERY),
            )  # fmt: skip
            top.set_xlabel("clinic minutes (5 s per query)")
        axes[r, 0].legend()
    fig.tight_layout()
    fig.savefig(cfg["output_png"], dpi=120)

    with Path(cfg["target_csv"]).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(targets[0]))
        writer.writeheader()
        writer.writerows(targets)
    print(f"wrote {out_csv}, {cfg['target_csv']}, {cfg['output_png']}")


if __name__ == "__main__":
    main()
