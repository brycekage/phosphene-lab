"""Phase 6: assemble the pilot as a static web page (HTML, CSS, JS, and its pictures).

The page in ``app/pilot_site/`` runs entirely in the browser, so it can be hosted for free
as a static Hugging Face Space. This copies it next to the pre-rendered stimuli and writes
``theme.css`` from the shared design tokens, so it keeps the demo's look.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from phosphene_lab.demo.theme import COLORS

PAGE_FILES = ("index.html", "pilot.css", "pilot.js", "session.js")


def theme_css() -> str:
    tokens = "\n".join(f"  --ph-{k.replace('_', '-')}: {v};" for k, v in COLORS.items())
    header = "/* Generated from phosphene_lab/demo/theme.py; do not edit. */"
    return f"{header}\n:root {{\n{tokens}\n}}\n"


def build_site(source_dir: str | Path, stimuli_dir: str | Path, out: str | Path) -> Path:
    """Write the page and every picture the manifest names into ``out`` (replaced)."""
    source, stimuli, out = Path(source_dir), Path(stimuli_dir), Path(out)
    manifest = json.loads((stimuli / "manifest.json").read_text(encoding="utf-8"))
    pictures = sorted({t[key] for t in manifest["trials"] for key in ("image", "original")})
    for path in pictures:
        if not (stimuli / path).is_file():
            raise FileNotFoundError(f"manifest names {stimuli / path}, which does not exist")

    if out.exists():
        shutil.rmtree(out)
    (out / "stimuli").mkdir(parents=True)
    for name in PAGE_FILES:
        shutil.copy2(source / name, out / name)
    (out / "theme.css").write_text(theme_css(), encoding="utf-8")
    shutil.copy2(stimuli / "manifest.json", out / "stimuli" / "manifest.json")
    for path in pictures:
        (out / "stimuli" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stimuli / path, out / "stimuli" / path)
    return out
