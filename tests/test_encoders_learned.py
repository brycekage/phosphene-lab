"""Tests for E3, the learned encoder trained through the differentiable simulator."""

import dataclasses

import pytest
import torch

from phosphene_lab.encoders.learned import (
    E3TrainConfig,
    LearnedEncoder,
    balanced_mse,
    load_e3,
    rasterize,
    train_learned_encoder,
)
from phosphene_lab.eval.observer import Observer
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.phosphene import PhospheneMap, PhospheneRenderer, RenderConfig


def _cross_map():
    return PhospheneMap(
        x_deg=torch.tensor([-5.0, 5.0, 0.0, 0.0]),
        y_deg=torch.tensor([0.0, 0.0, 4.0, -4.0]),
        threshold_uA=torch.full((4,), 20.0),
    )


def _grid_map(n):
    return PhospheneMap.from_array(ElectrodeArray.grid(n))


# --- rasterize ------------------------------------------------------------------------


def test_rasterize_puts_each_phosphene_at_its_pixel():
    # Only the right phosphene carries a value; the box spans x, y in [-5, 5] deg.
    raster = rasterize(_cross_map(), torch.tensor([0.0, 1.0, 0.0, 0.0]), (16, 16), blur_px=0)
    assert raster.shape == (1, 1, 16, 16)
    assert raster.sum() == pytest.approx(1.0)
    row, col = divmod(int(raster.argmax()), 16)
    assert (row, col) == (8, 15)  # x = +5 deg is the right edge, y = 0 the middle row


def test_rasterize_skips_dead_phosphenes():
    pmap = _cross_map().with_alive(torch.tensor([True, True, True, False]))
    assert rasterize(pmap, torch.ones(4), (16, 16), blur_px=0).sum() == pytest.approx(3.0)


# --- LearnedEncoder -------------------------------------------------------------------


@pytest.mark.parametrize("n", [64, 1024])
def test_one_network_serves_any_electrode_count(n):
    torch.manual_seed(0)
    encoder = LearnedEncoder()
    stim = encoder(torch.rand(2, 3, 24, 24), _grid_map(n))
    assert stim.shape == (2, n)
    assert stim.min() >= 0 and stim.max() <= encoder.config.max_current_uA


def test_large_images_are_shrunk_before_the_network():
    torch.manual_seed(0)
    encoder = LearnedEncoder(max_side=32)
    image = torch.rand(1, 3, 96, 128)
    assert encoder.field(image, _grid_map(64)).shape[-2:] == (24, 32)
    assert encoder(image, _grid_map(64)).shape == (1, 64)


def test_output_depends_on_the_patient_map():
    torch.manual_seed(0)
    encoder = LearnedEncoder()
    image = torch.rand(1, 3, 24, 24)
    pmap = _grid_map(256)
    high = dataclasses.replace(pmap, threshold_uA=torch.full((256,), 80.0))
    assert not torch.allclose(encoder(image, pmap), encoder(image, high))


def test_dead_electrodes_stay_off():
    torch.manual_seed(0)
    alive = torch.ones(256, dtype=torch.bool)
    alive[::2] = False
    stim = LearnedEncoder()(torch.rand(1, 3, 24, 24), _grid_map(256).with_alive(alive))
    assert torch.all(stim[0, ~alive] == 0)


def test_gradients_reach_the_network_through_the_renderer():
    torch.manual_seed(0)
    encoder, pmap = LearnedEncoder(), _grid_map(64)
    renderer = PhospheneRenderer(pmap, RenderConfig(resolution=32))
    renderer.render(encoder(torch.rand(2, 3, 24, 24), pmap)).mean().backward()
    grads = [p.grad for p in encoder.parameters()]
    assert all(g is not None for g in grads) and sum(g.abs().sum() for g in grads) > 0


# --- training -------------------------------------------------------------------------


def test_balanced_mse_makes_drawing_nothing_expensive_on_sparse_targets():
    target = torch.zeros(4, 32, 32)
    target[:, 10:14, 10:14] = 1.0
    blank = torch.zeros_like(target)
    assert torch.nn.functional.mse_loss(blank, target) < 0.02  # why plain MSE collapsed
    assert balanced_mse(blank, target) == pytest.approx(0.5)  # half the loss: lit pixels missed
    assert balanced_mse(target, target) == 0.0


def test_balanced_mse_equals_plain_mse_when_nothing_is_lit():
    pred = torch.rand(2, 8, 8)
    target = torch.zeros(2, 8, 8)
    assert balanced_mse(pred, target) == pytest.approx(
        torch.nn.functional.mse_loss(pred, target).item()
    )


def test_training_reduces_the_loss():
    g = torch.Generator().manual_seed(0)
    images = torch.zeros(16, 3, 16, 16)
    labels = torch.randint(0, 2, (16,), generator=g)
    for i, label in enumerate(labels):  # class 0: left bar, class 1: right bar
        images[i, :, 3:13, (3 if label == 0 else 10) : (6 if label == 0 else 13)] = 1.0
    cfg = E3TrainConfig(
        steps=60, batch_size=8, lr=3e-3, electrode_counts=(64,), render=RenderConfig(resolution=32)
    )
    _, _, history = train_learned_encoder(images, images, labels, n_classes=2, cfg=cfg)
    first = sum(h["loss"] for h in history[:10]) / 10
    last = sum(h["loss"] for h in history[-10:]) / 10
    assert last < 0.8 * first


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    encoder = LearnedEncoder(width=8)
    observer = Observer(3)
    path = tmp_path / "e3.pt"
    torch.save(
        {"encoder": encoder.state_dict(), "observer": observer.state_dict(), "width": 8,
         "max_side": 64, "n_classes": 3},
        path,
    )  # fmt: skip
    loaded, loaded_observer = load_e3(str(path), RenderConfig())
    image, pmap = torch.rand(1, 3, 16, 16), _grid_map(64)
    torch.testing.assert_close(loaded(image, pmap), encoder.eval()(image, pmap))
    assert loaded_observer.head[-1].out_features == 3
