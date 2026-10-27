"""Phase 6 pilot: sighted volunteers name letters and spot obstacles in simulated percepts.

The stimuli are rendered once by ``scripts/make_pilot_stimuli.py`` (so the page needs no
datasets or networks) and listed in ``manifest.json``. The page that shows them is static
(``app/pilot_site/``); its session rules live in ``session.js``. Nothing is stored.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Trial:
    task: str
    encoder: str
    label: str
    image: str  # percept PNG, relative to the stimuli folder
    original: str  # the camera image, shown only after practice answers
    practice: bool = False


def choose_images(
    labels: np.ndarray,
    encoders: Sequence[str],
    classes: Iterable[int],
    per_class: int,
    rng: np.random.Generator,
    exclude: Iterable[int] = (),
) -> dict[str, list[int]]:
    """Indices of ``per_class`` images of each class for every encoder, never one twice.

    Every encoder gets the same classes, so their scores are comparable, but different
    pictures, so a volunteer never sees the same image twice.
    """
    excluded = set(exclude)
    picks: dict[str, list[int]] = {e: [] for e in encoders}
    for c in classes:
        pool = [int(i) for i in np.flatnonzero(labels == c) if int(i) not in excluded]
        need = per_class * len(encoders)
        if len(pool) < need:
            raise ValueError(f"class {c} has {len(pool)} usable images, need {need}")
        chosen = rng.choice(pool, size=need, replace=False).tolist()
        for k, enc in enumerate(encoders):
            picks[enc] += chosen[k * per_class : (k + 1) * per_class]
    return picks


def load_trials(stimuli_dir: str | Path) -> tuple[list[Trial], dict[str, list[str]]]:
    manifest = json.loads((Path(stimuli_dir) / "manifest.json").read_text(encoding="utf-8"))
    return [Trial(**t) for t in manifest["trials"]], manifest["classes"]
