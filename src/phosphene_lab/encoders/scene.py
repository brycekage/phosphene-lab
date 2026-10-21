"""E2 scene-simplification encoder: show what matters for a task, drop texture and background.

Two pretrained networks describe the scene:

* Depth Anything V2 Small (monocular relative depth) gives "nearness" in [0, 1] per image.
* SegFormer-B0 trained on ADE20K labels each pixel (wall, floor, door, person, chair, ...).
  ADE20K is used instead of torchvision's VOC-trained DeepLabV3 because VOC has no wall, floor
  or door classes, which the navigation mode needs.

``simplify`` turns those maps into a sparse target image, by mode:

* navigation: near obstacles filled (brighter = nearer), floor boundaries and doors outlined,
  walls/floor/ceiling interiors dark.
* people: people filled, near obstacles faintly outlined for context.
* reading: foreground strokes only (whichever polarity is the minority), ignoring depth.

The encoder then samples the target at each phosphene. The same targets serve as the
reconstruction goal for the learned encoder E3.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from phosphene_lab.encoders.base import Encoder, luminance, sample_at_phosphenes
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig

# ADE20K label ids (0-indexed, as in the SegFormer config) that the modes refer to.
ADE = {
    "wall": 0, "sky": 2, "floor": 3, "ceiling": 5, "road": 6, "grass": 9, "sidewalk": 11,
    "person": 12, "door": 14, "table": 15, "chair": 19, "rug": 28, "skyscraper": 48,
    "path": 52, "stairs": 53, "screen door": 58, "stairway": 59,
}  # fmt: skip
WALKABLE = ("floor", "road", "grass", "sidewalk", "rug", "path")
BACKGROUND = ("wall", "sky", "ceiling", "skyscraper")
DOORS = ("door", "screen door")
HAZARDS = ("stairs", "stairway")
MODES = ("navigation", "people", "reading")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# (images) -> (nearness (B, 1, H, W) in [0, 1], ADE20K labels (B, H, W))
SceneModels = Callable[[Tensor], tuple[Tensor, Tensor]]


def _mask(seg: Tensor, names: tuple[str, ...]) -> Tensor:
    ids = torch.tensor([ADE[n] for n in names], device=seg.device)
    return torch.isin(seg, ids).unsqueeze(1).float()


def _outline(mask: Tensor) -> Tensor:
    """Pixels whose 3x3 neighborhood contains both inside and outside (a 2-px boundary)."""
    dilated = F.max_pool2d(mask, 3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, 3, stride=1, padding=1)
    return dilated - eroded


def simplify(
    images: Tensor, nearness: Tensor, seg: Tensor, mode: str, near_threshold: float = 0.5
) -> Tensor:
    """Task-specific target image in [0, 1]. Returns (B, 1, H, W)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
    if mode == "reading":
        lum = luminance(images)
        mean = lum.mean(dim=(2, 3), keepdim=True)
        bright = lum > mean
        bright_is_minority = bright.float().mean(dim=(2, 3), keepdim=True) < 0.5
        return torch.where(bright_is_minority, bright, lum < mean).float()

    walkable, background = _mask(seg, WALKABLE), _mask(seg, BACKGROUND)
    things = (1 - walkable) * (1 - background)
    near_things = things * (nearness >= near_threshold).float()
    if mode == "people":
        person = _mask(seg, ("person",))
        return torch.maximum(person * (0.6 + 0.4 * nearness), 0.3 * _outline(near_things))
    layers = [
        near_things * nearness,
        _mask(seg, HAZARDS),
        0.7 * _outline(walkable),
        _outline(_mask(seg, DOORS)),
    ]
    return torch.stack(layers).amax(dim=0).clamp(0.0, 1.0)


class PretrainedSceneModels:
    """Depth Anything V2 Small + SegFormer-B0 (ADE20K), run at reduced resolution on CPU."""

    def __init__(
        self,
        depth_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
        seg_id: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        depth_height: int = 252,  # Depth Anything needs multiples of its 14-px patch size
        seg_height: int = 256,
        device: str = "cpu",
    ) -> None:
        from transformers import AutoModelForDepthEstimation, SegformerForSemanticSegmentation

        self.depth = AutoModelForDepthEstimation.from_pretrained(depth_id).eval().to(device)
        self.seg = SegformerForSemanticSegmentation.from_pretrained(seg_id).eval().to(device)
        self.depth_height, self.seg_height, self.device = depth_height, seg_height, device

    @torch.no_grad()
    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor]:
        h, w = images.shape[-2:]
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        x = (images.to(self.device) - mean) / std

        dh = self.depth_height
        dw = max(14, round(dh * w / h / 14) * 14)
        d_in = F.interpolate(x, size=(dh, dw), mode="bilinear", align_corners=False)
        disparity = self.depth(pixel_values=d_in).predicted_depth.unsqueeze(1)
        disparity = F.interpolate(disparity, size=(h, w), mode="bilinear", align_corners=False)
        lo = disparity.amin(dim=(2, 3), keepdim=True)
        hi = disparity.amax(dim=(2, 3), keepdim=True)
        nearness = (disparity - lo) / (hi - lo).clamp(min=1e-6)  # relative: 1 = nearest

        sh = self.seg_height
        s_in = F.interpolate(x, size=(sh, round(sh * w / h)), mode="bilinear", align_corners=False)
        logits = self.seg(pixel_values=s_in).logits
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return nearness.cpu(), logits.argmax(dim=1).cpu()


class SceneEncoder(Encoder):
    def __init__(
        self,
        mode: str = "navigation",
        models: SceneModels | None = None,
        threshold: float = 0.3,
        on_current_uA: float = 60.0,
        config: RenderConfig = RenderConfig(),  # noqa: B008 - frozen dataclass
    ) -> None:
        super().__init__(config)
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose one of {MODES}")
        self.mode = mode
        self._models = models
        self.threshold = threshold
        self.on_current_uA = on_current_uA

    @property
    def models(self) -> SceneModels:
        if self._models is None:  # load the ~115 MB of weights only when first needed
            self._models = PretrainedSceneModels()
        return self._models

    def target(self, images: Tensor) -> Tensor:
        """The simplified scene this encoder tries to draw. (B, 1, H, W)."""
        if self.mode == "reading":  # depth and labels are unused; skip the networks
            b, _, h, w = images.shape
            nearness, seg = images.new_zeros(b, 1, h, w), images.new_zeros(b, h, w).long()
        else:
            nearness, seg = self.models(images)
        return simplify(images, nearness.to(images), seg.to(images.device), self.mode)

    def stimulate(self, images: Tensor, pmap: PhospheneMap) -> Tensor:
        value = sample_at_phosphenes(self.target(images), pmap)[:, 0]
        return torch.where(value > self.threshold, value, 0.0) * self.on_current_uA
