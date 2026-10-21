"""Tests for E2 scene simplification, using stub depth/segmentation models."""

import pytest
import torch

from phosphene_lab.encoders.scene import ADE, SceneEncoder, simplify
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig

H = W = 32


def _room():
    """Wall on top, floor below, a near chair in the middle, a far table on the right."""
    seg = torch.full((1, H, W), ADE["wall"])
    seg[:, H // 2 :] = ADE["floor"]
    near = torch.full((1, 1, H, W), 0.1)
    near[..., H // 2 :, :] = torch.linspace(0.3, 0.8, H - H // 2).view(-1, 1)  # floor recedes
    seg[:, 12:28, 10:18] = ADE["chair"]
    near[..., 12:28, 10:18] = 0.9
    seg[:, 14:20, 26:30] = ADE["table"]
    near[..., 14:20, 26:30] = 0.2
    return torch.zeros(1, 3, H, W), near, seg


def test_navigation_highlights_near_obstacles():
    images, near, seg = _room()
    target = simplify(images, near, seg, "navigation")[0, 0]
    assert target[18:26, 12:16].min() > 0.5  # chair interior
    assert target[15:19, 27:29].max() < 0.1  # far table interior is suppressed


def test_navigation_suppresses_wall_and_floor_interiors():
    images, near, seg = _room()
    target = simplify(images, near, seg, "navigation")[0, 0]
    assert target[2:8, 2:8].max() == 0  # wall
    assert target[26:32, 22:26].max() == 0  # floor away from any boundary


def test_navigation_outlines_the_floor_boundary():
    images, near, seg = _room()
    target = simplify(images, near, seg, "navigation")[0, 0]
    assert target[H // 2 - 1 : H // 2 + 1, 22:26].min() > 0  # wall meets floor


def test_navigation_outlines_doors():
    images, near, seg = _room()
    seg[:, 2:15, 22:28] = ADE["door"]
    target = simplify(images, near, seg, "navigation")[0, 0]
    assert target[2:15, 21:23].max() > 0.5  # door edge
    assert target[6:10, 24:26].max() == 0  # door interior


def test_people_mode_prefers_people_over_furniture():
    images, near, seg = _room()
    seg[:, 4:14, 22:28] = ADE["person"]
    near[..., 4:14, 22:28] = 0.4  # person farther than the chair
    target = simplify(images, near, seg, "people")[0, 0]
    assert target[6:12, 23:27].min() > 0.5
    assert target[18:26, 12:16].max() < target[6:12, 23:27].min()


def test_reading_mode_ignores_contrast_polarity():
    images = torch.zeros(1, 3, H, W)
    images[..., 8:24, 14:18] = 1.0  # white bar on black
    near, seg = torch.zeros(1, 1, H, W), torch.zeros(1, H, W, dtype=torch.long)
    a = simplify(images, near, seg, "reading")
    b = simplify(1 - images, near, seg, "reading")
    torch.testing.assert_close(a, b)
    assert a[0, 0, 10, 15] == 1 and a[0, 0, 2, 2] == 0


def test_unknown_mode_is_an_error():
    images, near, seg = _room()
    with pytest.raises(ValueError, match="mode"):
        simplify(images, near, seg, "dancing")


class _StubModels:
    def __call__(self, images):
        _, near, seg = _room()
        b = images.shape[0]
        return near.expand(b, -1, -1, -1), seg.expand(b, -1, -1)


def test_scene_encoder_runs_with_injected_models():
    pmap = PhospheneMap.from_array(ElectrodeArray.grid(256))
    encoder = SceneEncoder(models=_StubModels(), config=RenderConfig(max_active=40))
    stim = encoder(torch.zeros(2, 3, H, W), pmap)
    assert stim.shape == (2, 256)
    assert 0 < (stim[0] > 0).sum() <= 40
    assert stim.max() <= encoder.config.max_current_uA


@pytest.mark.slow
def test_pretrained_models_produce_aligned_maps():
    from phosphene_lab.encoders.scene import PretrainedSceneModels

    near, seg = PretrainedSceneModels()(torch.rand(2, 3, 60, 80))
    assert near.shape == (2, 1, 60, 80) and seg.shape == (2, 60, 80)
    assert 0 <= near.min() and near.max() <= 1
    assert seg.dtype == torch.long and seg.max() < 150
