"""Shared encoder plumbing: place an image on the visual field and read it at each phosphene.

Every encoder maps an RGB frame plus a patient's phosphene map to one current per electrode.
The image is "fitted to the cloud": it is stretched (keeping its aspect ratio) over the
smallest visual-field box that contains every alive phosphene, so the whole picture lands
where the implant can actually draw. Sampling uses bilinear ``grid_sample``, so encoders built
on it stay differentiable end to end.

Images are (B, C, H, W) with row 0 at the top of the visual field and column 0 at the left,
matching the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig, enforce_limits


@dataclass(frozen=True)
class FieldBox:
    """Visual-field rectangle (deg) that an image is stretched over."""

    x_min_deg: float
    x_max_deg: float
    y_min_deg: float
    y_max_deg: float


def fit_image_to_cloud(pmap: PhospheneMap, image_hw: tuple[int, int]) -> FieldBox:
    """Smallest box with the image's aspect ratio that contains every alive phosphene."""
    if not pmap.alive.any():
        raise ValueError("no alive phosphenes: cannot place an image on an empty map")
    x, y = pmap.x_deg[pmap.alive], pmap.y_deg[pmap.alive]
    x_lo, x_hi, y_lo, y_hi = (float(v) for v in (x.min(), x.max(), y.min(), y.max()))
    width, height = x_hi - x_lo, y_hi - y_lo
    if width == height == 0.0:  # a single alive phosphene: give it a 1-deg-wide image
        width = 1.0
    aspect = image_hw[1] / image_hw[0]
    # Grow the short side around its center until the box matches the image's aspect.
    if width < height * aspect:
        width = height * aspect
    else:
        height = width / aspect
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    return FieldBox(cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2)


def sample_at_phosphenes(images: Tensor, pmap: PhospheneMap) -> Tensor:
    """Bilinear image values at each phosphene. (B, C, H, W) -> (B, C, N)."""
    box = fit_image_to_cloud(pmap, images.shape[-2:])
    # grid_sample wants coordinates in [-1, 1] with -1 at the left/top edge of the image.
    gx = 2 * (pmap.x_deg - box.x_min_deg) / (box.x_max_deg - box.x_min_deg) - 1
    gy = 2 * (box.y_max_deg - pmap.y_deg) / (box.y_max_deg - box.y_min_deg) - 1
    grid = torch.stack([gx, gy], dim=-1).to(images)
    grid = grid.expand(images.shape[0], 1, -1, -1)  # (B, 1, N, 2)
    # Border padding: phosphenes on the box edge read the edge pixel, not a half-black blend.
    values = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return values[:, :, 0]


class Encoder(nn.Module):
    """Base class: subclasses implement ``stimulate``; ``forward`` applies hardware limits."""

    def __init__(self, config: RenderConfig = RenderConfig()) -> None:  # noqa: B008
        super().__init__()
        self.config = config

    def stimulate(self, images: Tensor, pmap: PhospheneMap) -> Tensor:
        """Raw currents (B, N) in uA, before limits."""
        raise NotImplementedError

    def forward(self, images: Tensor, pmap: PhospheneMap) -> Tensor:
        stim = self.stimulate(images, pmap) * pmap.alive.to(images)
        return enforce_limits(stim, self.config.max_current_uA, self.config.max_active)
