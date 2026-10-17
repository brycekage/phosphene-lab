"""Tests for the differentiable phosphene renderer (PROJECT_SPEC Phase 1)."""

import torch

from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.phosphene import (
    PhospheneMap,
    PhospheneRenderer,
    RenderConfig,
    enforce_limits,
)


def _single(x_deg, y_deg=0.0, threshold=20.0):
    return PhospheneMap(
        x_deg=torch.tensor([x_deg]),
        y_deg=torch.tensor([y_deg]),
        threshold_uA=torch.tensor([threshold]),
    )


def _grid_renderer(n=256, **cfg):
    pmap = PhospheneMap.from_array(ElectrodeArray.grid(n))
    return PhospheneRenderer(pmap, RenderConfig(**cfg)), pmap


# --- the six tests required by the spec ---------------------------------------------


def test_zero_stimulation_renders_black():
    renderer, pmap = _grid_renderer()
    image = renderer.render(torch.zeros(len(pmap)))
    assert image.shape == (128, 128)
    assert image.max().item() == 0.0


def test_dead_electrodes_never_light():
    pmap = PhospheneMap.from_array(ElectrodeArray.grid(256))
    alive = torch.ones(len(pmap), dtype=torch.bool)
    alive[::3] = False
    renderer = PhospheneRenderer(pmap.with_alive(alive), RenderConfig())
    stim = torch.zeros(len(pmap))
    stim[~alive] = 100.0
    assert renderer.render(stim).max().item() == 0.0


def test_phosphene_size_increases_with_eccentricity():
    cfg = RenderConfig(resolution=256)

    def spot_area(ecc):
        image = PhospheneRenderer(_single(ecc), cfg).render(torch.tensor([100.0]))
        return (image > 0.1 * image.max()).sum().item()

    assert spot_area(8.0) > spot_area(2.0)
    near, far = PhospheneRenderer(_single(2.0), cfg), PhospheneRenderer(_single(8.0), cfg)
    current = torch.tensor([[100.0]])
    assert far.sigma_deg(current).item() > near.sigma_deg(current).item()


def test_gradients_flow_from_percept_to_stimulation():
    renderer, pmap = _grid_renderer()
    stim = torch.full((len(pmap),), 40.0, requires_grad=True)
    renderer.render(stim).sum().backward()
    assert torch.isfinite(stim.grad).all()
    assert (stim.grad.abs() > 0).all()


def test_gradients_are_finite_at_zero_stimulation():
    renderer, pmap = _grid_renderer()
    stim = torch.zeros(len(pmap), requires_grad=True)
    renderer.render(stim).sum().backward()
    assert torch.isfinite(stim.grad).all()


def test_adaptation_reduces_brightness_over_repeated_frames():
    renderer, pmap = _grid_renderer()
    stim = torch.full((1, 60, len(pmap)), 80.0)
    frames = renderer.render_sequence(stim)
    assert frames.shape == (1, 60, 128, 128)
    means = frames.mean(dim=(-2, -1))[0]
    assert (torch.diff(means) < 0).all()
    assert torch.allclose(frames[0, 0], renderer.render(stim[0, 0]))
    assert means[-1] < 0.8 * means[0]


def test_adaptation_recovers_after_stimulation_stops():
    renderer, pmap = _grid_renderer(adaptation_tau_s=0.5)
    stim = torch.zeros(1, 120, len(pmap))
    stim[:, :30] = 80.0
    stim[:, -1] = 80.0  # probe after 89 silent frames (~3 s at 30 Hz)
    frames = renderer.render_sequence(stim)
    adapted, recovered = frames[0, 29].mean(), frames[0, -1].mean()
    assert recovered > adapted
    assert torch.allclose(frames[0, -1], frames[0, 0], rtol=0.05)


def test_same_seed_gives_identical_output():
    def run(seed):
        pmap = PhospheneMap.from_array(ElectrodeArray.thread_insertion(256, seed=seed))
        stim = torch.linspace(0, 100, len(pmap))
        return PhospheneRenderer(pmap, RenderConfig()).render(stim)

    assert torch.equal(run(4), run(4))
    assert not torch.equal(run(4), run(5))


# --- additional behavior ------------------------------------------------------------


def test_brightness_increases_with_current_and_is_bounded():
    renderer = PhospheneRenderer(_single(4.0), RenderConfig())
    peaks = [renderer.render(torch.tensor([c])).max().item() for c in (10.0, 25.0, 60.0, 100.0)]
    assert peaks == sorted(peaks) and peaks[0] < peaks[-1]
    assert 0.0 <= min(peaks) and max(peaks) <= 1.0


def test_neighboring_phosphenes_merge():
    pmap = PhospheneMap(
        x_deg=torch.tensor([6.0, 6.4]),
        y_deg=torch.tensor([0.0, 0.0]),
        threshold_uA=torch.tensor([20.0, 20.0]),
    )
    renderer = PhospheneRenderer(pmap, RenderConfig(resolution=256))
    both = renderer.render(torch.tensor([100.0, 100.0]))
    one = renderer.render(torch.tensor([100.0, 0.0]))
    row, col = renderer.pixel_index(6.2, 0.0)
    assert both[row, col] > one[row, col]


def test_batched_render_matches_single():
    renderer, pmap = _grid_renderer()
    stim = torch.rand(3, len(pmap)) * 100
    batched = renderer.render(stim)
    assert batched.shape == (3, 128, 128)
    assert torch.allclose(batched[1], renderer.render(stim[1]), atol=1e-6)


def test_enforce_limits_caps_current_and_active_count():
    stim = torch.tensor([[150.0, 10.0, 90.0, 0.0, 50.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
    stim.requires_grad_(True)
    out = enforce_limits(stim, max_current_uA=100.0, max_active=2)
    assert out.max().item() <= 100.0
    assert ((out > 0).sum(dim=1) <= 2).all()
    assert out[0].tolist() == [100.0, 0.0, 90.0, 0.0, 0.0]
    out.sum().backward()
    assert stim.grad[0, 2] == 1.0 and stim.grad[0, 1] == 0.0


def test_render_applies_limits_from_config():
    renderer, pmap = _grid_renderer(max_current_uA=100.0)
    capped = renderer.render(torch.full((len(pmap),), 100.0))
    over = renderer.render(torch.full((len(pmap),), 500.0))
    assert torch.equal(capped, over)
