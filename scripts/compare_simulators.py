"""Phase 1: cross-check our phosphene renderer against dynaphos.

Both simulators get the same phosphene locations (from the retinotopic map of one electrode
array), the same field of view, and the same current on every electrode. Writes a figure
(the two percepts plus phosphene size vs eccentricity) and a CSV of agreement metrics.

Usage:
    python scripts/compare_simulators.py --config configs/phase1_compare.yaml [--quick]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from phosphene_lab.config import load_config  # noqa: E402
from phosphene_lab.simulator.array import ElectrodeArray  # noqa: E402
from phosphene_lab.simulator.phosphene import (  # noqa: E402
    PhospheneMap,
    PhospheneRenderer,
    RenderConfig,
)
from phosphene_lab.simulator.reference import load_dynaphos_simulator_at  # noqa: E402


def build_array(cfg: dict, seed: int) -> ElectrodeArray:
    n = cfg["n_electrodes"]
    if cfg["layout"] == "grid":
        return ElectrodeArray.grid(n)
    return ElectrodeArray.thread_insertion(n, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_compare.yaml")
    parser.add_argument("--quick", action="store_true", help="small sizes for CI smoke tests")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    torch.manual_seed(cfg["seed"])

    array = build_array(cfg["array"], cfg["seed"])
    pmap = PhospheneMap.from_array(array)
    current = cfg["current_uA"]

    ours = PhospheneRenderer(pmap, RenderConfig(**cfg["render"]))
    our_image = ours.render(torch.full((len(pmap),), current))
    our_sigma = ours.sigma_deg(torch.full((1, len(pmap)), current))[0]

    dyn = load_dynaphos_simulator_at(
        pmap.x_deg.numpy(), pmap.y_deg.numpy(), cfg["dynaphos"], cfg["seed"]
    )
    if dyn.num_phosphenes != len(pmap):
        raise RuntimeError("dynaphos dropped phosphenes outside its field of view")
    amplitude = torch.full((dyn.num_phosphenes,), current * 1e-6)  # dynaphos uses amperes
    for _ in range(cfg["dynaphos"]["n_frames"]):
        dyn_image = dyn(amplitude)
    dyn_image = dyn_image.flip(0)  # dynaphos puts the bottom of the field in row 0
    dyn_sigma = dyn.sigma.get().flatten()

    ecc = torch.hypot(pmap.x_deg, pmap.y_deg)
    sigma_rel_diff = ((our_sigma - dyn_sigma).abs() / dyn_sigma).max().item()
    pixel_corr = np.corrcoef(our_image.flatten().numpy(), dyn_image.flatten().numpy())[0, 1]
    metrics = {
        "n_electrodes": len(pmap),
        "current_uA": current,
        "pixel_correlation": round(float(pixel_corr), 4),
        "sigma_max_relative_difference": round(sigma_rel_diff, 4),
        "ours_mean_brightness": round(our_image.mean().item(), 4),
        "dynaphos_mean_brightness": round(dyn_image.mean().item(), 4),
    }

    out_csv = Path(cfg["output_csv"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(metrics.items())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    extent = [-cfg["render"]["field_deg"] / 2, cfg["render"]["field_deg"] / 2] * 2
    for ax, image, title in zip(
        axes[:2],
        [our_image, dyn_image],
        ["phosphene-lab (ours)", "dynaphos (uniform thresholds)"],
        strict=True,
    ):
        ax.imshow(image.detach().numpy(), cmap="gray", vmin=0, vmax=1, extent=extent)
        ax.set_title(f"{title}\n{len(pmap)} electrodes, {current:g} uA", fontsize=10)
        ax.set_xlabel("x (deg)")
    axes[0].set_ylabel("y (deg)")
    order = torch.argsort(ecc)
    axes[2].plot(ecc[order], dyn_sigma[order], lw=3, alpha=0.5, label="dynaphos")
    axes[2].plot(ecc[order], our_sigma[order].detach(), ls="--", label="ours")
    axes[2].set_xlabel("eccentricity (deg)")
    axes[2].set_ylabel("phosphene sigma (deg)")
    axes[2].set_title(f"size vs eccentricity at {current:g} uA", fontsize=10)
    axes[2].legend()
    fig.tight_layout()
    out_png = Path(cfg["output_png"])
    fig.savefig(out_png, dpi=150)

    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"wrote {out_png} and {out_csv}")


if __name__ == "__main__":
    main()
