"""Phase 4: rerun the benchmark for E2/E3 with the map the system believes after calibration.

For each virtual patient, the encoders work from: (a) the true map, (b) the uncalibrated
retinotopic guess (every electrode assumed alive, at the population-median threshold), and
(c) the active-calibration estimate after each query budget. Percepts are always rendered
with the true map. The result is task accuracy as a function of calibration effort.

Usage:
    python scripts/closed_loop.py --config configs/phase4_closed_loop.yaml [--quick]
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from phosphene_lab.calibration.active import ActiveCalibrator
from phosphene_lab.calibration.common import SECONDS_PER_QUERY, CalibrationConfig, map_errors
from phosphene_lab.config import load_config
from phosphene_lab.eval.benchmark import (
    build_encoders,
    evaluate_cell,
    load_task_data,
    score_with_observer,
)
from phosphene_lab.eval.tasks import TASKS
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4_closed_loop.yaml")
    parser.add_argument("--quick", action="store_true", help="one seed, one task, small data")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    torch.manual_seed(cfg["seed"])
    render = RenderConfig(**cfg["render"])
    params = PatientParams(**cfg["patient_model"])
    array = ElectrodeArray.grid(cfg["n_electrodes"])

    # Calibrate each patient once; every task and encoder reuses the same estimated maps.
    patients, beliefs = {}, {}
    for seed in cfg["seeds"]:
        patient = VirtualPatient(array, params, seed=cfg["patient_seed_offset"] + seed)
        calibrator = ActiveCalibrator(array, CalibrationConfig.from_population(params), seed)
        snapshots = calibrator.run(patient, cfg["calibration_queries"])
        nominal = PhospheneMap.from_array(array, threshold_uA=params.threshold_median_uA)
        truth = patient.true_phosphene_map()
        maps = {"true": truth, "uncalibrated": nominal}
        maps |= {f"active_{q}": m for q, m in snapshots}
        patients[seed], beliefs[seed] = patient, maps
        for name, m in maps.items():
            print(f"seed {seed} {name}: {map_errors(m, truth)}")

    rows, start = [], time.perf_counter()
    for task_name in cfg["tasks"]:
        data = load_task_data(task_name, cfg["data_root"], cfg["limits"][task_name])
        encoders = build_encoders(
            cfg["encoders"], TASKS[task_name].mode, task_name, render, cfg["checkpoint_dir"]
        )
        for seed in cfg["seeds"]:
            patient = patients[seed]
            truth = patient.true_phosphene_map()
            for enc_name, encoder in encoders.items():
                # "accuracy": an observer retrained on this map's percepts (fully adapted).
                # "fixed_accuracy": the observer trained under the true map, reading this
                # map's percepts (not adapted). The "true" map comes first in `beliefs`.
                fixed_observer = None
                for map_name, belief in beliefs[seed].items():
                    result, observer = evaluate_cell(
                        data, encoder, patient, render, cfg["observer_epochs"], seed,
                        encoder_map=belief, observer_width=cfg["observer_width"],
                        return_observer=True,
                    )  # fmt: skip
                    if map_name == "true":
                        fixed_observer = observer
                    fixed = score_with_observer(
                        fixed_observer, data, encoder, patient, render, belief
                    )
                    queries = int(map_name.split("_")[1]) if map_name.startswith("active") else 0
                    rows.append(
                        {"task": task_name, "encoder": enc_name, "map": map_name, "seed": seed,
                         "queries": queries,
                         "clinic_minutes": round(queries * SECONDS_PER_QUERY / 60, 1)}
                        | {k: round(v, 4) for k, v in result.items()}
                        | {"fixed_accuracy": round(fixed, 4)}
                        | {k: round(v, 4) for k, v in map_errors(belief, truth).items()}
                    )  # fmt: skip
            print(f"{task_name} seed {seed}: {time.perf_counter() - start:.0f} s elapsed")

    out_csv = Path(cfg["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups: dict[tuple, list[dict]] = {}  # insertion order keeps the map order
    for r in rows:
        groups.setdefault((r["task"], r["encoder"], r["map"]), []).append(r)
    summary = []
    for (task_name, enc_name, map_name), members in sorted(
        groups.items(), key=lambda kv: (kv[0][0], kv[0][1])
    ):
        entry = {"task": task_name, "encoder": enc_name, "map": map_name, "n_seeds": len(members)}
        for key in ("accuracy", "fixed_accuracy"):
            v = np.array([m[key] for m in members])
            entry[f"{key}_mean"] = round(float(v.mean()), 4)
            entry[f"{key}_std"] = round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4)
        summary.append(entry)
    with Path(cfg["summary_csv"]).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {out_csv} and {cfg['summary_csv']}")


if __name__ == "__main__":
    main()
