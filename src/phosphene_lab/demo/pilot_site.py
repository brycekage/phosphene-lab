"""Phase 6: assemble the pilot as a static web page (HTML, CSS, JS, and its pictures).

The page in ``app/pilot_site/`` runs entirely in the browser, so it can be hosted for free
as a static Hugging Face Space. The build writes ONE ``index.html`` with the stylesheets and
scripts inside it, next to the pre-rendered stimuli. It has to: a private Space opens the
page with a signed token in its URL and admits later requests only when their referrer is
that signed page, so a script imported by another script is refused (401) and the page never
starts (the pictures also carry the token; see ``assetUrl`` in ``session.js``). The CSS
colors come from the shared design tokens, so the page keeps the demo's look.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from phosphene_lab.demo.theme import COLORS

# The tags in the source index.html that the build replaces with inline copies.
STYLE_TAGS = (
    '<link rel="stylesheet" href="theme.css">',
    '<link rel="stylesheet" href="pilot.css">',
)
SCRIPT_TAG = '<script type="module" src="pilot.js"></script>'


def theme_css() -> str:
    tokens = "\n".join(f"  --ph-{k.replace('_', '-')}: {v};" for k, v in COLORS.items())
    header = "/* Generated from phosphene_lab/demo/theme.py; do not edit. */"
    return f"{header}\n:root {{\n{tokens}\n}}\n"


def bundle_page(source_dir: str | Path) -> str:
    """index.html with theme + page CSS and session.js + pilot.js inlined (one module)."""
    source = Path(source_dir)
    html = (source / "index.html").read_text(encoding="utf-8")
    css = theme_css() + (source / "pilot.css").read_text(encoding="utf-8")
    session = (source / "session.js").read_text(encoding="utf-8")
    page = (source / "pilot.js").read_text(encoding="utf-8")
    # One module instead of two: drop session.js's exports and pilot.js's import of them.
    session = re.sub(r"^export ", "", session, flags=re.MULTILINE)
    page, n = re.subn(r'^import \{[^}]*\} from "\./session\.js";\n', "", page, flags=re.MULTILINE)
    script = session + "\n" + page
    if n != 1 or "</script" in script or "</style" in css:
        raise ValueError("pilot page sources cannot be inlined as they are")
    for tag in (*STYLE_TAGS, SCRIPT_TAG):
        if tag not in html:
            raise ValueError(f"index.html no longer contains {tag}")
    html = html.replace(STYLE_TAGS[0], f"<style>\n{css}</style>").replace(STYLE_TAGS[1], "")
    return html.replace(SCRIPT_TAG, f'<script type="module">\n{script}</script>')


def build_site(source_dir: str | Path, stimuli_dir: str | Path, out: str | Path) -> Path:
    """Write the one-file page and every picture the manifest names into ``out`` (replaced)."""
    source, stimuli, out = Path(source_dir), Path(stimuli_dir), Path(out)
    manifest = json.loads((stimuli / "manifest.json").read_text(encoding="utf-8"))
    pictures = sorted({t[key] for t in manifest["trials"] for key in ("image", "original")})
    for path in pictures:
        if not (stimuli / path).is_file():
            raise FileNotFoundError(f"manifest names {stimuli / path}, which does not exist")

    if out.exists():
        shutil.rmtree(out)
    (out / "stimuli").mkdir(parents=True)
    (out / "index.html").write_text(bundle_page(source), encoding="utf-8")
    shutil.copy2(stimuli / "manifest.json", out / "stimuli" / "manifest.json")
    for path in pictures:
        (out / "stimuli" / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stimuli / path, out / "stimuli" / path)
    return out
