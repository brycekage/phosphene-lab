"""Tests for the shared encoder plumbing: image placement and sampling (PROJECT_SPEC Phase 2)."""

import pytest
import torch

from phosphene_lab.encoders.base import (
    Encoder,
    fit_image_to_cloud,
    image_to_field,
    sample_at_phosphenes,
)
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig


def _cross_map(top=4.0):
    """Four well-separated phosphenes: left, right, top, bottom."""
    return PhospheneMap(
        x_deg=torch.tensor([-5.0, 5.0, 0.0, 0.0]),
        y_deg=torch.tensor([0.0, 0.0, top, -top]),
        threshold_uA=torch.full((4,), 20.0),
    )


def _grid_map(n=256):
    return PhospheneMap.from_array(ElectrodeArray.grid(n))


class _Passthrough(Encoder):
    """Current proportional to the sampled grayscale value."""

    def stimulate(self, images, pmap):
        return sample_at_phosphenes(images, pmap).mean(dim=1) * self.config.max_current_uA


# --- fit_image_to_cloud ---------------------------------------------------------------


def test_box_contains_every_alive_phosphene():
    pmap = _grid_map()
    box = fit_image_to_cloud(pmap, (64, 64))
    assert box.x_min_deg <= pmap.x_deg.min() and pmap.x_deg.max() <= box.x_max_deg
    assert box.y_min_deg <= pmap.y_deg.min() and pmap.y_deg.max() <= box.y_max_deg


@pytest.mark.parametrize("hw", [(64, 64), (48, 96), (120, 40)])
def test_box_matches_image_aspect(hw):
    box = fit_image_to_cloud(_cross_map(), hw)
    width = box.x_max_deg - box.x_min_deg
    height = box.y_max_deg - box.y_min_deg
    assert width / height == pytest.approx(hw[1] / hw[0])


def test_box_is_tight_on_the_long_axis_and_centered_on_the_short_one():
    box = fit_image_to_cloud(_cross_map(), (64, 64))  # cloud is 10 x 8 deg
    assert (box.x_min_deg, box.x_max_deg) == pytest.approx((-5.0, 5.0))
    assert (box.y_min_deg, box.y_max_deg) == pytest.approx((-5.0, 5.0))


def test_dead_phosphenes_do_not_stretch_the_box():
    pmap = _cross_map()
    pmap = PhospheneMap(
        x_deg=torch.cat([pmap.x_deg, torch.tensor([40.0])]),
        y_deg=torch.cat([pmap.y_deg, torch.tensor([0.0])]),
        threshold_uA=torch.full((5,), 20.0),
        alive=torch.tensor([True, True, True, True, False]),
    )
    assert fit_image_to_cloud(pmap, (64, 64)).x_max_deg == pytest.approx(5.0)


def test_single_alive_phosphene_gets_a_finite_box():
    pmap = _cross_map().with_alive(torch.tensor([False, True, False, False]))
    box = fit_image_to_cloud(pmap, (32, 64))
    assert box.x_max_deg > box.x_min_deg and box.y_max_deg > box.y_min_deg
    values = sample_at_phosphenes(torch.ones(1, 1, 32, 64), pmap)
    assert torch.isfinite(values).all()


def test_no_alive_phosphenes_is_an_error():
    pmap = _cross_map().with_alive(torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError, match="alive"):
        fit_image_to_cloud(pmap, (64, 64))


# --- sample_at_phosphenes -------------------------------------------------------------


def test_constant_image_samples_its_value_everywhere():
    values = sample_at_phosphenes(torch.full((2, 3, 32, 32), 0.5), _grid_map())
    assert values.shape == (2, 3, 256)
    torch.testing.assert_close(values, torch.full_like(values, 0.5))


def test_single_bright_pixel_drives_only_its_phosphene():
    image = torch.zeros(1, 1, 64, 64)
    image[0, 0, 32, 63] = 1.0  # right edge, just below the horizontal meridian
    values = sample_at_phosphenes(image, _cross_map())[0, 0]
    assert values[1] > 0
    assert torch.all(values[[0, 2, 3]] == 0)


def test_top_row_of_image_is_top_of_visual_field():
    image = torch.zeros(1, 1, 64, 64)
    image[0, 0, 0, :] = 1.0
    values = sample_at_phosphenes(image, _cross_map(top=5.0))[0, 0]
    assert values[2] > 0  # phosphene at the top edge of the box
    assert values[3] == 0  # phosphene at the bottom edge


# --- image_to_field -------------------------------------------------------------------


def test_image_to_field_fills_the_box_and_nothing_else():
    pmap = _cross_map()  # box spans x, y in [-5, 5] deg for a square image
    field = image_to_field(torch.ones(1, 1, 16, 16), pmap, RenderConfig(resolution=40))
    assert field.shape == (1, 1, 40, 40)
    # 20 deg over 40 px: 0.5 deg per pixel, so the box covers the middle 20 x 20 pixels.
    assert field[0, 0, 11:29, 11:29].min() == pytest.approx(1.0)
    assert field[0, 0, :9].max() == 0 and field[0, 0, :, 31:].max() == 0


def test_image_to_field_keeps_orientation():
    image = torch.zeros(1, 1, 16, 16)
    image[0, 0, :8, :8] = 1.0  # top-left quadrant
    field = image_to_field(image, _cross_map(), RenderConfig(resolution=40))[0, 0]
    assert field[12, 12] == pytest.approx(1.0) and field[27, 27] == 0.0


# --- Encoder base ---------------------------------------------------------------------


def test_encoder_output_shape():
    stim = _Passthrough()(torch.rand(3, 3, 32, 32), _grid_map())
    assert stim.shape == (3, 256)


def test_encoder_respects_current_and_power_limits():
    encoder = _Passthrough(RenderConfig(max_current_uA=60.0, max_active=10))
    stim = encoder(torch.rand(2, 3, 32, 32) * 5, _grid_map())  # oversaturated input
    assert stim.min() >= 0 and stim.max() <= 60.0
    assert ((stim > 0).sum(dim=1) <= 10).all()


def test_encoder_never_drives_dead_electrodes():
    alive = torch.ones(256, dtype=torch.bool)
    alive[::4] = False
    stim = _Passthrough()(torch.ones(1, 3, 32, 32), _grid_map().with_alive(alive))
    assert torch.all(stim[0, ~alive] == 0)
    assert torch.all(stim[0, alive] > 0)


def test_gradients_flow_to_the_image():
    image = torch.rand(1, 3, 32, 32, requires_grad=True)
    _Passthrough()(image * 0.5, _grid_map(64)).sum().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0


def test_base_encoder_is_abstract():
    with pytest.raises(NotImplementedError):
        Encoder()(torch.zeros(1, 1, 8, 8), _cross_map())
