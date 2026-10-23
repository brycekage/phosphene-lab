"""Tests for the Phase 5 demo logic (no Gradio, no downloaded networks)."""

import numpy as np
import pytest
import torch

from phosphene_lab.demo import Viewer, ViewSettings
from phosphene_lab.encoders.scene import ADE


def fake_scene_models(images):
    """Nearness rising left to right; a 'person' block in the middle."""
    b, _, h, w = images.shape
    nearness = torch.linspace(0, 1, w).expand(b, 1, h, w).clone()
    seg = torch.full((b, h, w), ADE["wall"])
    seg[:, h // 4 : 3 * h // 4, w // 3 : 2 * w // 3] = ADE["person"]
    return nearness, seg


def _photo(h=60, w=80):
    rng = np.random.default_rng(0)
    image = (rng.random((h, w, 3)) * 60).astype(np.uint8)
    image[15:45, 30:50] = 230  # a bright block
    return image


@pytest.fixture
def viewer(tmp_path):
    return Viewer(checkpoint_dir=tmp_path, scene_models=fake_scene_models, resolution=48)


@pytest.mark.parametrize("encoder", ["E0", "E1", "E2"])
def test_view_returns_three_aligned_images(viewer, encoder):
    out = viewer.view(_photo(), ViewSettings(n_electrodes=64, encoder=encoder))
    assert out.original.shape == (60, 80, 3)
    assert out.drawn.shape == out.percept.shape == (48, 48)
    for img in (out.drawn, out.percept):
        assert img.dtype == np.float32 and img.min() >= 0 and img.max() <= 1
    assert out.percept.max() > 0  # something is visible
    assert out.active_electrodes > 0 and out.total_current_uA > 0


def test_same_settings_give_the_same_percept(viewer):
    s = ViewSettings(n_electrodes=64, encoder="E0", seed=3)
    np.testing.assert_array_equal(
        viewer.view(_photo(), s).percept, viewer.view(_photo(), s).percept
    )


def test_dropout_removes_electrodes(viewer):
    image = np.full((60, 80, 3), 255, np.uint8)  # all bright: every working electrode is on
    full = viewer.view(image, ViewSettings(n_electrodes=256, encoder="E0", dropout=0.0))
    half = viewer.view(image, ViewSettings(n_electrodes=256, encoder="E0", dropout=0.5))
    assert half.active_electrodes < 0.7 * full.active_electrodes


def test_adaptation_dims_a_steady_image(viewer):
    fresh = viewer.view(_photo(), ViewSettings(n_electrodes=64, encoder="E0"))
    later = viewer.view(_photo(), ViewSettings(n_electrodes=64, encoder="E0", adaptation_s=3.0))
    assert later.percept.mean() < 0.9 * fresh.percept.mean()


def test_priority_mode_changes_what_e2_draws(viewer):
    nav = viewer.view(_photo(), ViewSettings(n_electrodes=64, encoder="E2", mode="navigation"))
    people = viewer.view(_photo(), ViewSettings(n_electrodes=64, encoder="E2", mode="people"))
    assert not np.allclose(nav.drawn, people.drawn)


def test_e3_is_offered_only_for_modes_it_was_trained_for(tmp_path):
    (tmp_path / "e3_obstacles.pt").touch()
    (tmp_path / "e3_letters.pt").touch()
    viewer = Viewer(checkpoint_dir=tmp_path, scene_models=fake_scene_models)
    assert "E3" in viewer.available_encoders("navigation")
    assert "E3" in viewer.available_encoders("reading")
    assert "E3" not in viewer.available_encoders("people")


def test_e3_is_offered_only_when_its_checkpoint_exists(viewer):
    assert "E3" not in viewer.available_encoders("navigation")
    with pytest.raises(ValueError, match="checkpoint"):
        viewer.view(_photo(), ViewSettings(n_electrodes=64, encoder="E3"))
