"""Phase 4: the pointing task, a direct test of the calibrated phosphene map.

For each virtual patient, E0 (naive) shows white dots using: (a) the true map, (b) the
uncalibrated retinotopic guess, and (c) the active-calibration estimate after each query
budget. Percepts are rendered with the true map, the patient points at the brightness
centroid, and the miss is measured in degrees. No observer is trained, so a wrong map cannot
be learned around.

Usage:
    python scripts/pointing_task.py --config configs/phase4_pointing.yaml [--quick]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from phosphene_lab.calibration.active import ActiveCalibrator
from phosphene_lab.calibration.common import SECONDS_PER_QUERY, CalibrationConfig, map_errors
from phosphene_lab.config import load_config
from phosphene_lab.encoders.naive import NaiveEncoder
from phosphene_lab.eval.pointing import dot_images, pointing_errors
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase4_pointing.yaml")
    parser.add_argument("--quick", action="store_true", help="one seed, small array")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    torch.manual_seed(cfg["seed"])
    render = RenderConfig(**cfg["render"])
    params = PatientParams(**cfg["patient_model"])
    array = ElectrodeArray.grid(cfg["n_electrodes"])
    encoder = NaiveEncoder(config=render)

    rows = []
    for seed in cfg["seeds"]:
        patient = VirtualPatient(array, params, seed=cfg["patient_seed_offset"] + seed)
        calibrator = ActiveCalibrator(array, CalibrationConfig.from_population(params), seed)
        snapshots = calibrator.run(patient, cfg["calibration_queries"])
        truth = patient.true_phosphene_map()
        maps = {"true": (0, truth)}
        maps["uncalibrated"] = (
            0, PhospheneMap.from_array(array, threshold_uA=params.threshold_median_uA)
        )  # fmt: skip
        maps |= {f"active_{q}": (q, m) for q, m in snapshots}
        images, _ = dot_images(cfg["n_dots"], cfg["image_size"], cfg["dot_radius_frac"], seed)
        for map_name, (queries, belief) in maps.items():
            errors = pointing_errors(encoder, images, belief, truth, render)
            seen = ~errors.isnan()
            rows.append(
                {"map": map_name, "seed": seed, "queries": queries,
                 "clinic_minutes": round(queries * SECONDS_PER_QUERY / 60, 1),
                 "mean_error_deg": round(errors[seen].mean().item(), 4),
                 "median_error_deg": round(errors[seen].median().item(), 4),
                 "seen_fraction": round(seen.float().mean().item(), 4)}
                | {k: round(v, 4) for k, v in map_errors(belief, truth).items()}
            )  # fmt: skip
            print(f"seed {seed} {map_name}: mean miss {rows[-1]['mean_error_deg']:.2f} deg")

    out_csv = Path(cfg["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for map_name in dict.fromkeys(r["map"] for r in rows):  # keeps the map order
        members = [r for r in rows if r["map"] == map_name]
        entry = {"map": map_name, "queries": members[0]["queries"], "n_seeds": len(members)}
        for key in ("mean_error_deg", "seen_fraction", "location_error_deg"):
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
