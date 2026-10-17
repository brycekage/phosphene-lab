"""Differentiable phosphene renderer: stimulation currents -> percept image.

Model (simplified from dynaphos, van der Grinten et al. 2024):

* Each electrode produces a Gaussian phosphene at its visual-field location.
* Size: current spreads over a cortical disk of radius sqrt(I / K) mm (Tehovnik's square-root
  rule; K is tissue excitability). Dividing by cortical magnification converts the disk to
  degrees, so the same current gives larger phosphenes in the periphery.
* Brightness: a sigmoid of current minus the electrode's threshold, shifted so zero current
  gives exactly zero brightness.
* Merging: overlapping phosphenes add, then saturate softly (1 - exp(-s * total)), so
  neighbors blend into one blob and gradients never vanish at a hard clip.
* Adaptation: a leaky charge trace per electrode (time constant tau) suppresses brightness
  under repeated stimulation and recovers when stimulation stops.
* Dead electrodes produce nothing. Hard limits cap the current per electrode and the number
  of simultaneously active electrodes (power budget).

Images are (H, W) with row 0 at the top of the visual field and column 0 at the left.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.retinotopy import (
    DEFAULT_RETINOTOPY,
    RetinotopyParams,
    magnification,
)


@dataclass(frozen=True)
class RenderConfig:
    resolution: int = 128  # pixels per side
    field_deg: float = 20.0  # width of the rendered visual field
    max_current_uA: float = 100.0
    max_active: int | None = None  # None: no power budget
    brightness_slope_uA: float = 5.0  # width of the brightness sigmoid
    excitability_uA_per_mm2: float = 675.0  # K in r = sqrt(I / K) (dynaphos default)
    radius_to_sigma: float = 0.5  # Gaussian sigma per activated-disk radius (dynaphos)
    saturation: float = 2.5  # soft-saturation gain for overlapping phosphenes
    frame_rate_hz: float = 30.0
    adaptation_tau_s: float = 1.0  # recovery time constant of the charge trace
    adaptation_scale_uA_s: float = 100.0  # trace level that dims brightness by 1/e


@dataclass(frozen=True)
class PhospheneMap:
    """Per-electrode phosphene parameters: where each one appears and how it responds."""

    x_deg: Tensor
    y_deg: Tensor
    threshold_uA: Tensor
    size_scale: Tensor | None = None
    alive: Tensor | None = None

    def __post_init__(self) -> None:
        n = len(self.x_deg)
        if self.size_scale is None:
            object.__setattr__(self, "size_scale", torch.ones(n))
        if self.alive is None:
            object.__setattr__(self, "alive", torch.ones(n, dtype=torch.bool))

    def __len__(self) -> int:
        return len(self.x_deg)

    def with_alive(self, alive: Tensor) -> PhospheneMap:
        return dataclasses.replace(self, alive=alive)

    @classmethod
    def from_array(
        cls,
        array: ElectrodeArray,
        threshold_uA: float = 20.0,
        retinotopy: RetinotopyParams = DEFAULT_RETINOTOPY,
    ) -> PhospheneMap:
        """Nominal map predicted from retinotopy alone (what an uncalibrated system assumes)."""
        x, y = array.to_visual_field(retinotopy)
        return cls(x_deg=x, y_deg=y, threshold_uA=torch.full((len(array),), threshold_uA))


def enforce_limits(stim: Tensor, max_current_uA: float, max_active: int | None) -> Tensor:
    """Clamp currents to [0, max] and keep only the ``max_active`` strongest per sample."""
    stim = stim.clamp(0.0, max_current_uA)
    if max_active is None or max_active >= stim.shape[-1]:
        return stim
    keep = torch.zeros_like(stim, dtype=torch.bool)
    keep.scatter_(-1, stim.topk(max_active, dim=-1).indices, True)
    return stim * keep


class PhospheneRenderer(nn.Module):
    def __init__(
        self,
        pmap: PhospheneMap,
        config: RenderConfig = RenderConfig(),  # noqa: B008 - frozen dataclass
        retinotopy: RetinotopyParams = DEFAULT_RETINOTOPY,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_buffer("x_deg", pmap.x_deg.float())
        self.register_buffer("y_deg", pmap.y_deg.float())
        self.register_buffer("threshold_uA", pmap.threshold_uA.float())
        self.register_buffer("size_scale", pmap.size_scale.float())
        self.register_buffer("alive", pmap.alive.float())
        ecc = torch.hypot(self.x_deg, self.y_deg)
        self.register_buffer("magnification", magnification(ecc, retinotopy))

        half = config.field_deg / 2
        self.pixel_deg = config.field_deg / config.resolution
        centers = torch.linspace(
            -half + self.pixel_deg / 2, half - self.pixel_deg / 2, config.resolution
        )
        self.register_buffer("pixel_x", centers)
        self.register_buffer("pixel_y", centers.flip(0))  # row 0 is the top of the field

    def pixel_index(self, x_deg: float, y_deg: float) -> tuple[int, int]:
        """(row, col) of the pixel containing a visual-field point."""
        half = self.config.field_deg / 2
        col = int((x_deg + half) // self.pixel_deg)
        row = int((half - y_deg) // self.pixel_deg)
        return row, col

    def brightness(self, current_uA: Tensor) -> Tensor:
        """Per-electrode brightness in [0, 1); exactly 0 at zero current or dead electrodes."""
        slope = self.config.brightness_slope_uA
        floor = torch.sigmoid(-self.threshold_uA / slope)
        b = (torch.sigmoid((current_uA - self.threshold_uA) / slope) - floor) / (1 - floor)
        return b * self.alive

    def sigma_deg(self, current_uA: Tensor) -> Tensor:
        """Physical phosphene sigma (deg) before the pixel blur floor."""
        cfg = self.config
        # +1 uA keeps the square root differentiable at zero current.
        radius_mm = torch.sqrt((current_uA.clamp(min=0) + 1.0) / cfg.excitability_uA_per_mm2)
        return cfg.radius_to_sigma * radius_mm / self.magnification * self.size_scale

    def render(self, stim_uA: Tensor) -> Tensor:
        """Render one frame without adaptation. stim: (N,) or (B, N) -> (H, W) or (B, H, W)."""
        single = stim_uA.dim() == 1
        stim = stim_uA.unsqueeze(0) if single else stim_uA
        current = enforce_limits(stim, self.config.max_current_uA, self.config.max_active)
        image = self._compose(current, self.brightness(current))
        return image[0] if single else image

    def render_sequence(self, stim_uA: Tensor) -> Tensor:
        """Render frames with adaptation. stim: (B, T, N) -> (B, T, H, W)."""
        cfg = self.config
        dt = 1.0 / cfg.frame_rate_hz
        decay = torch.exp(torch.tensor(-dt / cfg.adaptation_tau_s))
        trace = torch.zeros_like(stim_uA[:, 0])
        frames = []
        for t in range(stim_uA.shape[1]):
            current = enforce_limits(stim_uA[:, t], cfg.max_current_uA, cfg.max_active)
            gain = torch.exp(-trace / cfg.adaptation_scale_uA_s)
            frames.append(self._compose(current, self.brightness(current) * gain))
            trace = trace * decay + current * dt
        return torch.stack(frames, dim=1)

    def _compose(self, current: Tensor, brightness: Tensor) -> Tensor:
        # Half-pixel blur floor: sub-pixel phosphenes still show up as single-pixel dots.
        sigma = torch.sqrt(self.sigma_deg(current) ** 2 + (self.pixel_deg / 2) ** 2)
        inv_two_var = 1 / (2 * sigma**2)  # (B, N)
        # Isotropic Gaussians are separable, so the image is a (B,H,N) @ (B,N,W) matmul
        # instead of an N x H x W stack.
        gx = torch.exp(-((self.pixel_x - self.x_deg[:, None]) ** 2) * inv_two_var[..., None])
        gy = torch.exp(-((self.pixel_y - self.y_deg[:, None]) ** 2) * inv_two_var[..., None])
        total = torch.bmm((gy * brightness[..., None]).transpose(1, 2), gx)
        return 1 - torch.exp(-self.config.saturation * total)
