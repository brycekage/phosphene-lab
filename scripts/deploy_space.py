"""Stage a Hugging Face Space and upload it (private by default).

A Gradio Space (the demo, ``configs/demo.yaml``) gets exactly what the app needs: the package
source, ``app/app.py``, the configs, the E3 checkpoints, a root ``app.py`` shim that puts
``src/`` on the path, a pinned ``requirements.txt`` (CPU torch), and a Space README. Hugging
Face hosts Gradio Spaces only with a paid PRO account.

A static Space (the pilot, ``configs/phase6_pilot.yaml``, ``sdk: static``) gets the folder
built by ``scripts/build_pilot_site.py`` and a README; static Spaces are free.

Everything is staged in a git-ignored folder first so you can inspect what will be uploaded.

Upload needs a Hugging Face write token (``hf auth login``).

Usage:
    python scripts/deploy_space.py [--config configs/demo.yaml] [--quick]

``--quick`` only stages the folder; nothing is uploaded.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import shutil
from pathlib import Path

from phosphene_lab.config import load_config

# Runtime dependencies pinned to the local versions (gradio comes from sdk_version).
PACKAGES = [
    "numpy", "torch", "torchvision", "matplotlib", "pyyaml", "opencv-python-headless",
    "pulse2percept", "dynaphos", "transformers", "h5py", "scikit-image", "botorch", "pillow",
]  # fmt: skip

SHIM = '''"""Hugging Face Space entry point: run {entry} with src/ on the import path."""

import runpy
import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root / "src"))
runpy.run_path(str(root / "{entry}"), run_name="__main__")
'''

README = """---
title: {title}
emoji: 👁
colorFrom: indigo
colorTo: gray
{sdk_lines}
pinned: false
---

# {title}

{summary} Source and results: the phosphene-lab repository.
"""


def requirements() -> str:
    lines = ["--extra-index-url https://download.pytorch.org/whl/cpu"]
    for name in PACKAGES:
        version = md.version(name).split("+")[0]  # drop the local "+cpu" tag
        lines.append(f"{name}=={version}")
    return "\n".join(lines) + "\n"


def readme(cfg: dict) -> str:
    if cfg.get("sdk", "gradio") == "static":
        sdk_lines = "sdk: static\napp_file: index.html"
    else:
        sdk_lines = (f"sdk: gradio\nsdk_version: {cfg['sdk_version']}\n"
                     'python_version: "3.11"\napp_file: app.py')  # fmt: skip
    return README.format(sdk_lines=sdk_lines, title=cfg["title"], summary=cfg["summary"])


def stage(cfg: dict, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    if cfg.get("sdk", "gradio") == "static":
        site = Path(cfg["site_dir"])
        if not (site / "index.html").is_file():
            raise SystemExit(f"{site} has no index.html; run scripts/build_pilot_site.py first")
        shutil.copytree(site, out)
        (out / "README.md").write_text(readme(cfg), encoding="utf-8")
        return
    out.mkdir(parents=True)
    for item in cfg["copy"] + cfg["checkpoints"]:
        src, dst = Path(item), out / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    (out / "app.py").write_text(SHIM.format(entry=cfg["entry"]), encoding="utf-8")
    (out / "requirements.txt").write_text(requirements(), encoding="utf-8")
    (out / "README.md").write_text(readme(cfg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--quick", action="store_true", help="stage only, no upload")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)["space"]
    out = Path(cfg["staging_dir"])
    stage(cfg, out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"staged {out} ({size:.1f} MB)")
    if args.quick:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(
        cfg["repo_id"], repo_type="space", space_sdk=cfg.get("sdk", "gradio"),
        private=cfg["private"], exist_ok=True,
    )  # fmt: skip
    info = api.repo_info(cfg["repo_id"], repo_type="space")
    if cfg["private"] and not info.private:
        raise SystemExit(f"{cfg['repo_id']} already exists and is public; refusing to upload")
    # A static Space is exactly the built site: remove page files the build no longer makes.
    stale = ["*.html", "*.js", "*.css", "stimuli/**"] if cfg.get("sdk") == "static" else None
    api.upload_folder(folder_path=out, repo_id=cfg["repo_id"], repo_type="space",
                      delete_patterns=stale)  # fmt: skip
    print(f"uploaded to https://huggingface.co/spaces/{cfg['repo_id']} (private={info.private})")


if __name__ == "__main__":
    main()
