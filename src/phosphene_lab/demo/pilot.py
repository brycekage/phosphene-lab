"""Phase 6 pilot: sighted volunteers name letters and spot obstacles in simulated percepts.

The stimuli are rendered once by ``scripts/make_pilot_stimuli.py`` (so the page needs no
datasets or networks) and listed in ``manifest.json``. A session runs one block per task:
a short practice with the answer shown, then the scored trials of every encoder shuffled
together, so a volunteer cannot tell which encoder made a picture. Nothing is stored: the
session lives in the page and disappears when it closes.
"""

from __future__ import annotations

import json
import random
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


class PilotSession:
    """One volunteer's run through the blocks. ``answer`` scores and advances."""

    def __init__(self, trials: Sequence[Trial], blocks: Sequence[str], seed: int) -> None:
        rng = random.Random(seed)
        self.order: list[Trial] = []
        self._block_starts: set[int] = set()
        for task in blocks:
            self._block_starts.add(len(self.order))
            practice = [t for t in trials if t.task == task and t.practice]
            scored = [t for t in trials if t.task == task and not t.practice]
            rng.shuffle(scored)
            self.order += practice + scored
        self.position = 0
        self.responses: list[tuple[Trial, str, bool]] = []
        self._intro_shown: set[int] = set()

    @property
    def finished(self) -> bool:
        return self.position >= len(self.order)

    @property
    def current(self) -> Trial | None:
        return None if self.finished else self.order[self.position]

    @property
    def last(self) -> tuple[Trial, str, bool] | None:
        return self.responses[-1] if self.responses else None

    @property
    def at_block_start(self) -> bool:
        """True until the volunteer has seen the instructions for the block about to start."""
        return self.position in self._block_starts and self.position not in self._intro_shown

    def start_block(self) -> None:
        self._intro_shown.add(self.position)

    def answer(self, choice: str) -> bool:
        trial = self.current
        if trial is None:
            raise RuntimeError("the session is finished")
        correct = choice == trial.label
        self.responses.append((trial, choice, correct))
        self.position += 1
        return correct

    def progress(self) -> tuple[int, int]:
        """(scored trials answered, scored trials in total) for the current block."""
        task = (self.current or self.order[-1]).task
        block = [t for t in self.order if t.task == task and not t.practice]
        done = sum(1 for t, _, _ in self.responses if t.task == task and not t.practice)
        return done, len(block)

    def summary(self) -> dict[str, dict[str, tuple[int, int]]]:
        """task -> encoder -> (correct, answered), scored trials only."""
        out: dict[str, dict[str, list[int]]] = {}
        for trial, _, correct in self.responses:
            if trial.practice:
                continue
            cell = out.setdefault(trial.task, {}).setdefault(trial.encoder, [0, 0])
            cell[0] += int(correct)
            cell[1] += 1
        return {
            task: {enc: (c, n) for enc, (c, n) in sorted(cells.items())}
            for task, cells in out.items()
        }
