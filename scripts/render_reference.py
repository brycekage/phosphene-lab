"""Phase 0: render one percept from pulse2percept and one from dynaphos.

Usage:
    python scripts/render_reference.py --config configs/phase0_reference.yaml [--quick]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from phosphene_lab.simulator.reference import render_dynaphos, render_pulse2percept  # noqa: E402


def load_config(path: str, quick: bool) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if quick:
        for section, overrides in cfg.get("quick", {}).items():
            cfg[section].update(overrides)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase0_reference.yaml")
    parser.add_argument("--quick", action="store_true", help="small sizes for CI smoke tests")
    args = parser.parse_args()

    cfg = load_config(args.config, args.quick)
    p2p_image = render_pulse2percept(cfg["pulse2percept"])
    dyn_image = render_dynaphos(cfg["dynaphos"], cfg["seed"]).detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    titles = [
        f"pulse2percept ({cfg['pulse2percept']['implant']}, scoreboard)",
        f"dynaphos ({cfg['dynaphos']['n_phosphenes']} phosphenes)",
    ]
    for ax, image, title in zip(axes, [p2p_image, dyn_image], titles, strict=True):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
