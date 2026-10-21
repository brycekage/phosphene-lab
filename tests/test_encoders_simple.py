"""Tests for the hand-designed encoders E0 (naive brightness) and E1 (edges)."""

import torch

from phosphene_lab.encoders.base import fit_image_to_cloud
from phosphene_lab.encoders.edges import EdgeEncoder
from phosphene_lab.encoders.naive import NaiveEncoder
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig


def _grid_map(n=256):
    return PhospheneMap.from_array(ElectrodeArray.grid(n))


def _split_image(value_left, value_right, hw=(48, 48)):
    image = torch.full((1, 3, *hw), value_right)
    image[..., : hw[1] // 2] = value_left
    return image


def _box_center_x(pmap, hw=(48, 48)):
    box = fit_image_to_cloud(pmap, hw)
    return (box.x_min_deg + box.x_max_deg) / 2


# --- E0 naive -------------------------------------------------------------------------


def test_naive_bright_image_turns_every_electrode_on():
    encoder = NaiveEncoder(on_current_uA=60.0)
    stim = encoder(torch.ones(1, 3, 32, 32), _grid_map())
    torch.testing.assert_close(stim, torch.full_like(stim, 60.0))


def test_naive_dark_image_is_silent():
    stim = NaiveEncoder()(torch.zeros(2, 3, 32, 32), _grid_map())
    assert stim.abs().max() == 0


def test_naive_follows_brightness_layout():
    pmap = _grid_map()
    stim = NaiveEncoder()(_split_image(1.0, 0.0), pmap)[0]
    cx = _box_center_x(pmap)
    assert torch.all(stim[pmap.x_deg < cx - 1] > 0)
    assert torch.all(stim[pmap.x_deg > cx + 1] == 0)


def test_naive_uses_luminance_not_one_channel():
    green = torch.zeros(1, 3, 32, 32)
    green[:, 1] = 1.0  # luminance ~0.59 > 0.5 threshold
    blue = torch.zeros(1, 3, 32, 32)
    blue[:, 2] = 1.0  # luminance ~0.11
    assert NaiveEncoder()(green, _grid_map()).min() > 0
    assert NaiveEncoder()(blue, _grid_map()).max() == 0


def test_naive_respects_power_budget():
    encoder = NaiveEncoder(config=RenderConfig(max_active=20))
    stim = encoder(torch.ones(1, 3, 32, 32), _grid_map())
    assert (stim > 0).sum() == 20


# --- E1 edges -------------------------------------------------------------------------


def test_edges_of_a_uniform_image_are_silent():
    for value in (0.0, 0.4, 1.0):
        stim = EdgeEncoder()(torch.full((1, 3, 32, 32), value), _grid_map())
        assert stim.abs().max() == 0


def test_edges_light_only_near_the_boundary():
    # Put a vertical step exactly under one phosphene in the right hemifield (the box
    # center is the vertical meridian, where there are no phosphenes).
    pmap, hw = _grid_map(), (48, 48)
    box = fit_image_to_cloud(pmap, hw)
    x_edge = float(pmap.x_deg[pmap.x_deg > 0].median())
    col = round((x_edge - box.x_min_deg) / (box.x_max_deg - box.x_min_deg) * hw[1])
    image = torch.zeros(1, 3, *hw)
    image[..., :col] = 1.0
    stim = EdgeEncoder()(image, pmap)[0]
    near = (pmap.x_deg - x_edge).abs() < 0.3
    far = (pmap.x_deg - x_edge).abs() > 3.0
    assert near.any() and torch.all(stim[near] > 0)
    assert torch.all(stim[far] == 0)


def test_edges_ignore_contrast_polarity():
    pmap = _grid_map()
    a = EdgeEncoder()(_split_image(1.0, 0.0), pmap)
    b = EdgeEncoder()(_split_image(0.0, 1.0), pmap)
    torch.testing.assert_close(a, b)


def test_edges_are_differentiable():
    image = _split_image(0.8, 0.2).requires_grad_(True)
    EdgeEncoder(hard=False)(image, _grid_map(64)).sum().backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
