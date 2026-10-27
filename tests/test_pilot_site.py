import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from phosphene_lab.demo import pilot_site, theme

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "pilot_site"


@pytest.mark.skipif(shutil.which("node") is None, reason="needs Node.js")
def test_session_rules_in_node():
    test_file = ROOT / "tests" / "js" / "session.test.mjs"
    result = subprocess.run(
        ["node", "--test", str(test_file)], capture_output=True, text=True, cwd=ROOT
    )
    assert result.returncode == 0, result.stdout + result.stderr


def make_stimuli(folder: Path, missing: bool = False) -> Path:
    (folder / "letters").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(folder / "letters" / "a.png")
    if not missing:
        Image.new("RGB", (8, 8)).save(folder / "letters" / "a_original.png")
    trial = {"task": "letters", "encoder": "E0", "label": "a", "image": "letters/a.png",
             "original": "letters/a_original.png", "practice": False}  # fmt: skip
    manifest = {"classes": {"letters": ["a"]}, "trials": [trial]}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_build_copies_the_page_and_every_stimulus(tmp_path):
    out = tmp_path / "site"
    pilot_site.build_site(SOURCE, make_stimuli(tmp_path / "stimuli"), out)
    for name in ("index.html", "pilot.js", "session.js", "pilot.css", "theme.css"):
        assert (out / name).is_file(), name
    manifest = json.loads((out / "stimuli" / "manifest.json").read_text(encoding="utf-8"))
    for trial in manifest["trials"]:
        assert (out / "stimuli" / trial["image"]).is_file()
        assert (out / "stimuli" / trial["original"]).is_file()


def test_build_fails_loudly_on_a_missing_stimulus(tmp_path):
    with pytest.raises(FileNotFoundError, match="a_original.png"):
        pilot_site.build_site(SOURCE, make_stimuli(tmp_path / "s", missing=True), tmp_path / "o")


def test_theme_css_carries_the_shared_tokens(tmp_path):
    out = tmp_path / "site"
    pilot_site.build_site(SOURCE, make_stimuli(tmp_path / "stimuli"), out)
    css = (out / "theme.css").read_text(encoding="utf-8")
    for name, value in theme.COLORS.items():
        assert f"--ph-{name.replace('_', '-')}: {value}" in css


def test_page_stores_nothing_and_calls_no_other_site():
    html = (SOURCE / "index.html").read_text(encoding="utf-8")
    assert "not saved" in html
    for path in SOURCE.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"https?://", text), path.name
        for api in ("localStorage", "sessionStorage", "indexedDB", "document.cookie",
                    "XMLHttpRequest", "sendBeacon", "WebSocket"):  # fmt: skip
            assert api not in text, (path.name, api)


def test_stylesheet_uses_tokens_and_accessibility_rules():
    css = (SOURCE / "pilot.css").read_text(encoding="utf-8")
    assert not re.findall(r"#[0-9a-fA-F]{3,6}\b", css)
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
