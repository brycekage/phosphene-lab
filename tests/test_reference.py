"""Phase 0 smoke tests: both reference simulators install and render."""

from pathlib import Path

import numpy as np
import pytest
import torch

from phosphene_lab.simulator.reference import (
    load_dynaphos_simulator,
    load_dynaphos_simulator_at,
    render_dynaphos,
    render_pulse2percept,
)

P2P_CFG = {"implant": "Orion", "amplitude_uA": 80, "rho_um": 400, "field_deg": 5, "step_deg": 0.25}
DYN_CFG = {
    "params": str(Path(__file__).parents[1] / "configs" / "dynaphos_params.yaml"),
    "n_phosphenes": 64,
    "resolution": 64,
    "amplitude_A": 100e-6,
    "n_frames": 2,
}


def test_pulse2percept_renders_nonblank_image():
    image = render_pulse2percept(P2P_CFG)
    assert image.ndim == 2
    assert np.isfinite(image).all()
    assert image.max() == pytest.approx(1.0)


def test_dynaphos_renders_nonblank_image():
    image = render_dynaphos(DYN_CFG, seed=0)
    assert image.shape == (64, 64)
    assert torch.isfinite(image).all()
    assert image.max() > 0


def test_dynaphos_gradients_reach_stimulation():
    sim = load_dynaphos_simulator(DYN_CFG, seed=0)
    amplitude = torch.full((sim.num_phosphenes,), 100e-6, requires_grad=True)
    sim(amplitude).sum().backward()
    assert amplitude.grad is not None
    assert torch.isfinite(amplitude.grad).all()


def test_dynaphos_leaves_global_torch_state_untouched():
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    load_dynaphos_simulator(DYN_CFG, seed=0)
    assert torch.equal(torch.rand(3), expected)
    assert not torch.are_deterministic_algorithms_enabled()


def test_dynaphos_places_phosphenes_at_requested_locations():
    cfg = {**DYN_CFG, "resolution": 128, "field_deg": 20.0, "uniform_thresholds": True}
    sim = load_dynaphos_simulator_at(np.array([5.0]), np.array([5.0]), cfg, seed=0)
    for _ in range(5):
        image = sim(torch.full((1,), 100e-6))
    row, col = np.unravel_index(int(image.argmax()), tuple(image.shape))
    assert col == pytest.approx(96, abs=1)  # x = +5 deg
    assert row == pytest.approx(96, abs=1)  # dynaphos row 0 is the bottom: y = +5 deg is low


def test_dynaphos_same_seed_same_output():
    a = render_dynaphos(DYN_CFG, seed=3)
    b = render_dynaphos(DYN_CFG, seed=3)
    assert torch.equal(a, b)
