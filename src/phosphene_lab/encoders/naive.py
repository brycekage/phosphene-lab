"""E0 naive encoder: an electrode is on when the image is bright where its phosphene appears.

This is the "scoreboard" approach many early prosthesis studies used: downsample brightness to
the phosphene grid and threshold. It ignores what is in the scene, so textured or cluttered
images light up almost everywhere.
"""

from __future__ import annotations

from torch import Tensor

from phosphene_lab.encoders.base import Encoder, luminance, sample_at_phosphenes
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig


class NaiveEncoder(Encoder):
    def __init__(
        self,
        threshold: float = 0.5,
        on_current_uA: float = 60.0,
        config: RenderConfig = RenderConfig(),  # noqa: B008 - frozen dataclass
    ) -> None:
        super().__init__(config)
        self.threshold = threshold
        self.on_current_uA = on_current_uA

    def stimulate(self, images: Tensor, pmap: PhospheneMap) -> Tensor:
        brightness = sample_at_phosphenes(luminance(images), pmap)[:, 0]
        return (brightness > self.threshold).to(images) * self.on_current_uA
