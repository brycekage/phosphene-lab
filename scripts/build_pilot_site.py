"""Phase 6: put the static pilot page and its pre-rendered pictures in one folder.

Run ``scripts/make_pilot_stimuli.py`` first. The folder is the whole site: preview it with
``python -m http.server 7861 -d results/pilot_site`` and open http://127.0.0.1:7861, or
upload it with ``scripts/deploy_space.py --config configs/phase6_pilot.yaml``.

Usage:
    python scripts/build_pilot_site.py [--config configs/phase6_pilot.yaml] [--quick]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phosphene_lab.config import load_config
from phosphene_lab.demo.pilot_site import build_site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6_pilot.yaml")
    parser.add_argument("--quick", action="store_true", help="use the quick stimuli")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    out = build_site(Path("app/pilot_site"), cfg["stimuli_dir"], cfg["site_dir"])
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"wrote {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
