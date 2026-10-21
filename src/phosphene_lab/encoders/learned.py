"""E3 learned encoder: a small CNN trained end to end through the differentiable simulator.

Instead of hand-picking a rule, the network learns which electrodes to drive, and how hard, by
seeing the percepts its choices produce. Training minimizes three terms:

* reconstruction: the rendered percept should look like the E2 simplified scene,
* task: a jointly trained observer should recognize the stimulus from the percept,
* power: fewer and weaker pulses are better (charge, heat, battery).

The patient's phosphene map is an input, rasterized onto the image grid as two channels (where
phosphenes are, and how much current they need), and the network's output is a stimulation
*field* sampled at each phosphene. So one network works for any electrode count or layout.
Every training step draws a new virtual patient, so it has to generalize across patients
rather than memorize one. Following Granley et al. (NeurIPS 2022) and de Ruyter van
Steveninck et al. (2022), who trained encoders through differentiable phosphene models.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from phosphene_lab.encoders.base import (
    Encoder,
    fit_image_to_cloud,
    gaussian_blur,
    image_to_field,
    sample_at_phosphenes,
)
from phosphene_lab.eval.observer import Observer
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import DEFAULT_PATIENT, PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap, PhospheneRenderer, RenderConfig


def rasterize(
    pmap: PhospheneMap, values: Tensor, hw: tuple[int, int], blur_px: float = 1.0
) -> Tensor:
    """Splat one value per alive phosphene onto the image grid. Returns (1, 1, H, W)."""
    h, w = hw
    box = fit_image_to_cloud(pmap, hw)
    col = ((pmap.x_deg - box.x_min_deg) / (box.x_max_deg - box.x_min_deg) * w).long()
    row = ((box.y_max_deg - pmap.y_deg) / (box.y_max_deg - box.y_min_deg) * h).long()
    flat = row.clamp(0, h - 1) * w + col.clamp(0, w - 1)
    raster = torch.zeros(h * w).index_add_(0, flat, values.float() * pmap.alive.float())
    raster = raster.view(1, 1, h, w)
    return gaussian_blur(raster, blur_px) if blur_px > 0 else raster


class LearnedEncoder(Encoder):
    def __init__(
        self,
        width: int = 32,
        config: RenderConfig = RenderConfig(),  # noqa: B008 - frozen dataclass
        max_side: int = 64,
    ) -> None:
        super().__init__(config)
        # Images are shrunk so the long side is at most this many pixels: a few hundred
        # phosphenes cannot show finer detail, and it keeps training within the CPU budget.
        self.max_side = max_side

        def conv(c_in: int, c_out: int, dilation: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=dilation, dilation=dilation), nn.ReLU()
            )

        # Dilations 1-2-4-8 give a ~31-px receptive field without downsampling.
        self.net = nn.Sequential(
            conv(5, width), conv(width, width, 2), conv(width, width, 4), conv(width, width, 8),
            conv(width, width), nn.Conv2d(width, 1, 1),
        )  # fmt: skip

    def map_channels(self, pmap: PhospheneMap, hw: tuple[int, int]) -> Tensor:
        """(1, 2, H, W): phosphene density, and the local mean threshold as a fraction of max."""
        density = rasterize(pmap, torch.ones(len(pmap)), hw)
        threshold = rasterize(pmap, pmap.threshold_uA / self.config.max_current_uA, hw)
        mean_threshold = threshold / density.clamp(min=1e-3)
        return torch.cat([density / density.amax().clamp(min=1e-6), mean_threshold], dim=1)

    def field(self, images: Tensor, pmap: PhospheneMap) -> Tensor:
        """Stimulation field in [0, 1] over the (possibly shrunk) image. (B, 1, h, w)."""
        h, w = images.shape[-2:]
        if max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            size = (max(1, round(h * scale)), max(1, round(w * scale)))
            images = F.interpolate(images, size=size, mode="area")
        maps = self.map_channels(pmap, images.shape[-2:]).to(images)
        x = torch.cat([images, maps.expand(images.shape[0], -1, -1, -1)], dim=1)
        return torch.sigmoid(self.net(x))

    def stimulate(self, images: Tensor, pmap: PhospheneMap) -> Tensor:
        value = sample_at_phosphenes(self.field(images, pmap), pmap)[:, 0]
        return value * self.config.max_current_uA


def load_e3(path: str, config: RenderConfig) -> tuple[LearnedEncoder, Observer]:
    """Encoder and observer from a checkpoint written by scripts/train_e3.py."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    encoder = LearnedEncoder(ckpt["width"], config, ckpt["max_side"])
    encoder.load_state_dict(ckpt["encoder"])
    observer = Observer(ckpt["n_classes"])
    observer.load_state_dict(ckpt["observer"])
    return encoder.eval(), observer.eval()


@dataclass(frozen=True)
class E3TrainConfig:
    steps: int = 2000
    batch_size: int = 32
    lr: float = 1e-3
    electrode_counts: tuple[int, ...] = (256, 1024)
    task_weight: float = 0.1
    power_weight: float = 0.05
    width: int = 32
    max_side: int = 64
    render: RenderConfig = field(default_factory=lambda: RenderConfig(resolution=64))
    patient: PatientParams = DEFAULT_PATIENT
    seed: int = 0


def train_learned_encoder(
    images: Tensor, targets: Tensor, labels: Tensor, n_classes: int, cfg: E3TrainConfig
) -> tuple[LearnedEncoder, Observer, list[dict]]:
    """Train E3 and its observer jointly on random virtual patients.

    ``targets`` are the simplified scenes (B, 1, H, W) the percept should resemble. Returns
    the encoder, the observer, and per-step losses.
    """
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    encoder = LearnedEncoder(cfg.width, cfg.render, cfg.max_side)
    observer = Observer(n_classes)
    params = list(encoder.parameters()) + list(observer.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.lr)
    history = []
    encoder.train()
    observer.train()
    for step in range(cfg.steps):
        array = ElectrodeArray.grid(rng.choice(cfg.electrode_counts))
        patient = VirtualPatient(array, cfg.patient, seed=cfg.seed * 1_000_003 + step)
        pmap = patient.true_phosphene_map()
        renderer = PhospheneRenderer(pmap, cfg.render)
        idx = torch.randint(0, len(labels), (cfg.batch_size,))

        stim = encoder(images[idx], pmap)
        percept = renderer.render(stim)
        recon = F.mse_loss(percept, image_to_field(targets[idx], pmap, cfg.render)[:, 0])
        task = F.cross_entropy(observer(percept), labels[idx])
        power = (stim / cfg.render.max_current_uA).mean()
        loss = recon + cfg.task_weight * task + cfg.power_weight * power

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(
            {"step": step, "loss": loss.item(), "recon": recon.item(), "task": task.item(),
             "power": power.item(), "n_electrodes": len(pmap)}
        )  # fmt: skip
    return encoder.eval(), observer.eval(), history
