"""Tests for the wedge-dipole retinotopic map (visual field <-> V1 surface)."""

import numpy as np
import pytest
import torch

from phosphene_lab.simulator.retinotopy import (
    DEFAULT_RETINOTOPY,
    cortex_to_visual_field,
    magnification,
    visual_field_to_cortex,
)


def _random_field_points(n=500, seed=0, max_ecc=10.0):
    g = torch.Generator().manual_seed(seed)
    r = 0.1 + (max_ecc - 0.1) * torch.rand(n, generator=g, dtype=torch.float64)
    theta = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * torch.pi
    return r * torch.cos(theta), r * torch.sin(theta)


def test_fovea_maps_to_cortex_origin():
    cx, cy, _ = visual_field_to_cortex(torch.tensor([0.0]), torch.tensor([0.0]))
    assert cx.abs().item() < 1e-6
    assert cy.abs().item() < 1e-6


def test_round_trip_recovers_visual_field_in_both_hemifields():
    x, y = _random_field_points()
    cx, cy, hemi = visual_field_to_cortex(x, y)
    x2, y2 = cortex_to_visual_field(cx, cy, hemi)
    assert torch.allclose(x2, x, atol=1e-8)
    assert torch.allclose(y2, y, atol=1e-8)


def test_hemisphere_follows_side_of_visual_field():
    _, _, hemi = visual_field_to_cortex(torch.tensor([2.0, -2.0]), torch.tensor([1.0, 1.0]))
    assert hemi.tolist() == [1, -1]


def test_left_and_right_hemifields_are_mirror_images():
    x, y = _random_field_points()
    cx_r, cy_r, _ = visual_field_to_cortex(x.abs(), y)
    cx_l, cy_l, _ = visual_field_to_cortex(-x.abs(), y)
    assert torch.allclose(cx_r, cx_l)
    assert torch.allclose(cy_r, cy_l)


def test_cortical_distance_increases_with_eccentricity():
    ecc = torch.linspace(0.0, 10.0, 50, dtype=torch.float64)
    cx, _, _ = visual_field_to_cortex(ecc, torch.zeros_like(ecc))
    assert (torch.diff(cx) > 0).all()


def test_matches_dynaphos_wedge_dipole_mapping():
    from dynaphos.cortex_models import get_mapping_from_visual_field_to_cortex

    p = DEFAULT_RETINOTOPY
    dyn = get_mapping_from_visual_field_to_cortex(
        {"model": "wedge-dipole", "k": p.k, "a": p.a, "b": p.b, "alpha": p.alpha}
    )
    x, y = _random_field_points()
    x = x.abs()  # dynaphos maps one hemifield; compare on the right one
    expected = dyn(x.numpy() + 1j * y.numpy())
    cx, cy, _ = visual_field_to_cortex(x, y)
    np.testing.assert_allclose(cx.numpy(), expected.real, atol=1e-9)
    np.testing.assert_allclose(cy.numpy(), expected.imag, atol=1e-9)


def test_magnification_decreases_and_tracks_horton_hoyt():
    ecc = torch.linspace(0.5, 10.0, 40, dtype=torch.float64)
    m = magnification(ecc)
    assert (torch.diff(m) < 0).all()
    horton_hoyt = 17.3 / (ecc + 0.75)
    assert ((m - horton_hoyt).abs() / horton_hoyt).max() < 0.10


def test_magnification_is_slope_of_map_along_horizontal_meridian():
    ecc = torch.linspace(0.5, 10.0, 20, dtype=torch.float64, requires_grad=True)
    cx, _, _ = visual_field_to_cortex(ecc, torch.zeros_like(ecc))
    (slope,) = torch.autograd.grad(cx.sum(), ecc)
    assert torch.allclose(slope, magnification(ecc.detach()), rtol=1e-6)


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_gradients_are_finite(direction):
    x, y = _random_field_points(n=50)
    if direction == "forward":
        x.requires_grad_(True)
        y.requires_grad_(True)
        cx, cy, _ = visual_field_to_cortex(x, y)
        (cx.sum() + cy.sum()).backward()
        grads = [x.grad, y.grad]
    else:
        cx, cy, hemi = visual_field_to_cortex(x, y)
        cx = cx.detach().requires_grad_(True)
        cy = cy.detach().requires_grad_(True)
        x2, y2 = cortex_to_visual_field(cx, cy, hemi)
        (x2.sum() + y2.sum()).backward()
        grads = [cx.grad, cy.grad]
    for g in grads:
        assert g is not None and torch.isfinite(g).all()
