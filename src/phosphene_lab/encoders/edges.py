"""E1 edge encoder: an electrode is on when its phosphene falls on an object outline.

Outlines carry most of the shape information in far fewer phosphenes than filled regions.
Edges come from a Sobel filter on luminance, normalized per image, then blurred by about one
phosphene spacing so thin lines are not missed between sample points.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from phosphene_lab.encoders.base import (
    Encoder,
    gaussian_blur,
    luminance,
    sample_at_phosphenes,
)
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8


def edge_map(images: Tensor, blur_px: float = 1.0, min_contrast: float = 0.1) -> Tensor:
    """Edge strength in [0, 1]. (B, 3, H, W) -> (B, 1, H, W)."""
    gray = F.pad(luminance(images), (1, 1, 1, 1), mode="replicate")  # no fake border edges
    kernel = torch.stack([_SOBEL_X, _SOBEL_X.T]).unsqueeze(1).to(images)
    gx, gy = F.conv2d(gray, kernel).unbind(dim=1)
    # sqrt(0) has an infinite gradient, so add a tiny floor inside the root.
    magnitude = torch.sqrt(gx**2 + gy**2 + 1e-12).unsqueeze(1) - 1e-6
    if blur_px > 0:
        magnitude = gaussian_blur(magnitude, blur_px)
    # A low-contrast image keeps its faint edges faint instead of being stretched to 1.
    peak = magnitude.amax(dim=(2, 3), keepdim=True).clamp(min=min_contrast)
    return (magnitude / peak).clamp(0.0, 1.0)


class EdgeEncoder(Encoder):
    def __init__(
        self,
        threshold: float = 0.2,
        on_current_uA: float = 60.0,
        blur_px: float = 1.0,
        hard: bool = True,
        config: RenderConfig = RenderConfig(),  # noqa: B008 - frozen dataclass
    ) -> None:
        super().__init__(config)
        self.threshold = threshold
        self.on_current_uA = on_current_uA
        self.blur_px = blur_px
        self.hard = hard  # False: smooth sigmoid instead of a step, for gradient-based use

    def stimulate(self, images: Tensor, pmap: PhospheneMap) -> Tensor:
        edges = sample_at_phosphenes(edge_map(images, self.blur_px), pmap)[:, 0]
        if self.hard:
            on = (edges > self.threshold).to(images)
        else:
            on = torch.sigmoid((edges - self.threshold) / 0.05)
        return on * self.on_current_uA
