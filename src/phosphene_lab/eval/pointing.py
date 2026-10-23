"""Pointing task: the implant shows a dot, and the patient points at where they see it.

This tests the phosphene map directly, with no observer to learn around its errors. The
encoder works from the map the system *believes*; the percept is rendered with the *true* map,
so a wrong map draws the dot in the wrong place. The patient is modeled as pointing at the
percept's brightness centroid, and the miss is measured against where a perfectly calibrated
system would have drawn the dot (the same image placed with the true map). Even the true map
misses by a little, because a few hundred blurry phosphenes cannot draw a dot exactly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from phosphene_lab.encoders.base import Encoder, image_to_field
from phosphene_lab.simulator.phosphene import PhospheneMap, PhospheneRenderer, RenderConfig


def dot_images(
    n: int, size: int = 48, radius_frac: float = 0.08, seed: int = 0
) -> tuple[Tensor, Tensor]:
    """``n`` black images with one white disc each. Returns images (n, 3, size, size) and the
    disc centers (n, 2) as (u, v) fractions of width and height, kept a radius from the edge."""
    g = torch.Generator().manual_seed(seed)
    centers = radius_frac + (1 - 2 * radius_frac) * torch.rand(n, 2, generator=g)
    coords = (torch.arange(size) + 0.5) / size
    du = coords[None, None, :] - centers[:, 0, None, None]
    dv = coords[None, :, None] - centers[:, 1, None, None]
    discs = (du**2 + dv**2 <= radius_frac**2).float()
    return discs[:, None].expand(-1, 3, -1, -1).contiguous(), centers


def centroid_deg(fields: Tensor, config: RenderConfig) -> tuple[Tensor, Tensor, Tensor]:
    """Brightness-weighted centroid (x, y in degrees) of each (B, R, R) field, and whether
    anything was visible at all."""
    half, px = config.field_deg / 2, config.field_deg / config.resolution
    centers = torch.linspace(-half + px / 2, half - px / 2, config.resolution)
    total = fields.sum(dim=(1, 2))
    seen = total > 1e-6
    safe = total.clamp(min=1e-6)
    x = (fields.sum(dim=1) * centers).sum(dim=1) / safe
    y = (fields.sum(dim=2) * centers.flip(0)).sum(dim=1) / safe  # row 0 is the top
    return x, y, seen


@torch.no_grad()
def pointing_errors(
    encoder: Encoder,
    images: Tensor,
    belief: PhospheneMap,
    truth: PhospheneMap,
    config: RenderConfig,
) -> Tensor:
    """Miss distance (degrees) for each image; NaN where the patient saw nothing."""
    percepts = PhospheneRenderer(truth, config).render(encoder(images, belief))
    x, y, seen = centroid_deg(percepts, config)
    goal_x, goal_y, _ = centroid_deg(image_to_field(images, truth, config)[:, 0], config)
    error = torch.hypot(x - goal_x, y - goal_y)
    return torch.where(seen, error, torch.full_like(error, float("nan")))
