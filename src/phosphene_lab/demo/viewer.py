"""Demo viewer: one camera image in, three pictures out (photo, what is drawn, the percept).

Everything the Gradio app shows is computed here, so it can be tested without a browser. The
patient is a default virtual patient; the encoders use its true phosphene map (perfect
calibration), and the calibration game covers what happens when the map is not known.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from phosphene_lab.encoders.base import Encoder, image_to_field, luminance
from phosphene_lab.encoders.edges import EdgeEncoder, edge_map
from phosphene_lab.encoders.learned import LearnedEncoder, load_e3
from phosphene_lab.encoders.naive import NaiveEncoder
from phosphene_lab.encoders.scene import MODES, PretrainedSceneModels, SceneEncoder, SceneModels
from phosphene_lab.eval.benchmark import make_patient
from phosphene_lab.simulator.phosphene import PhospheneMap, PhospheneRenderer, RenderConfig

ELECTRODE_COUNTS = (64, 256, 1024, 1600)
ENCODERS = {"E0": "E0 naive", "E1": "E1 edges", "E2": "E2 scene", "E3": "E3 learned"}
# Which trained E3 serves each priority mode: one E3 was trained per benchmark task, on that
# task's E2 targets. No task used the "people" targets, so E3 is not offered there.
E3_TASK = {"navigation": "obstacles", "reading": "letters"}


@dataclass(frozen=True)
class ViewSettings:
    n_electrodes: int = 256
    dropout: float = 0.1
    adaptation_s: float = 0.0  # how long this image has been shown, steadily
    encoder: str = "E2"
    mode: str = "navigation"
    seed: int = 0


@dataclass(frozen=True)
class ViewResult:
    original: np.ndarray  # (H, W, 3) uint8, as given
    drawn: np.ndarray  # (R, R) float32 in [0, 1]: what the encoder draws, placed in the field
    percept: np.ndarray  # (R, R) float32 in [0, 1]
    active_electrodes: int
    total_current_uA: float


def to_tensor(image: np.ndarray, max_side: int) -> Tensor:
    """(H, W, 3|4) or (H, W) uint8 -> (1, 3, h, w) float in [0, 1], long side <= max_side."""
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    x = torch.from_numpy(np.ascontiguousarray(image[..., :3])).permute(2, 0, 1)[None].float() / 255
    h, w = x.shape[-2:]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        x = F.interpolate(x, size=(round(h * scale), round(w * scale)), mode="area")
    return x


class Viewer:
    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        scene_models: SceneModels | None = None,
        resolution: int = 128,
        max_side: int = 240,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.render = RenderConfig(resolution=resolution)
        self.max_side = max_side
        self._scene_models = scene_models
        self._patient = lru_cache(maxsize=8)(self._make_patient)
        self._e3 = lru_cache(maxsize=3)(self._load_e3)

    def _scene(self, images: Tensor) -> tuple[Tensor, Tensor]:
        if self._scene_models is None:  # ~115 MB of weights, loaded on first use
            self._scene_models = PretrainedSceneModels()
        return self._scene_models(images)

    def warm(self) -> None:
        """Load the scene networks now, so the first visitor does not wait ~20 s for them."""
        self._scene(torch.zeros(1, 3, 32, 32))

    def _make_patient(
        self, n: int, dropout: float, seed: int
    ) -> tuple[PhospheneMap, PhospheneRenderer]:
        pmap = make_patient(n, dropout, "default", seed).true_phosphene_map()
        return pmap, PhospheneRenderer(pmap, self.render)

    def _e3_path(self, mode: str) -> Path | None:
        return self.checkpoint_dir / f"e3_{E3_TASK[mode]}.pt" if mode in E3_TASK else None

    def _load_e3(self, mode: str) -> LearnedEncoder:
        return load_e3(str(self._e3_path(mode)), self.render)[0]

    def available_encoders(self, mode: str) -> list[str]:
        names = ["E0", "E1", "E2"]
        path = self._e3_path(mode)
        return names + ["E3"] if path is not None and path.exists() else names

    @torch.no_grad()
    def view(self, image: np.ndarray, s: ViewSettings) -> ViewResult:
        if s.mode not in MODES:
            raise ValueError(f"unknown mode {s.mode!r}; choose one of {MODES}")
        if s.encoder not in self.available_encoders(s.mode):
            path = self._e3_path(s.mode)
            raise ValueError(
                f"{s.encoder} is not available in {s.mode} mode: "
                + ("no E3 was trained for it" if path is None else f"no checkpoint at {path}")
            )
        x = to_tensor(image, self.max_side)
        pmap, renderer = self._patient(s.n_electrodes, s.dropout, s.seed)
        encoder: Encoder
        if s.encoder == "E0":
            encoder, drawn = NaiveEncoder(config=self.render), luminance(x)
            stim = encoder(x, pmap)
        elif s.encoder == "E1":
            encoder, drawn = EdgeEncoder(config=self.render), edge_map(x)
            stim = encoder(x, pmap)
        else:
            scene = SceneEncoder(s.mode, models=self._scene, config=self.render)
            drawn = scene.target(x)
            if s.encoder == "E2":
                stim = scene.encode_targets(drawn, pmap)
            else:
                e3 = self._e3(s.mode)
                stim = e3(e3.inputs(x, drawn), pmap)

        if s.adaptation_s > 0:
            frames = max(1, round(s.adaptation_s * self.render.frame_rate_hz))
            percept = renderer.render_sequence(stim[:, None].expand(-1, frames, -1))[0, -1]
        else:
            percept = renderer.render(stim)[0]
        placed = image_to_field(drawn, pmap, self.render)[0, 0].clamp(0, 1)
        return ViewResult(
            original=image,
            drawn=placed.numpy().astype(np.float32),
            percept=percept.clamp(0, 1).numpy().astype(np.float32),
            active_electrodes=int((stim > 0).sum()),
            total_current_uA=float(stim.sum()),
        )
