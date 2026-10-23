"""Tests for the calibration game: the user plays the patient."""

import pytest

from phosphene_lab.demo.game import CalibrationGame
from phosphene_lab.simulator.patient import PatientParams

CORRELATED = PatientParams(
    location_jitter_mm=0.3, placement_offset_mm=1.5, placement_rotation_deg=5.0,
    threshold_log_sd=0.2, threshold_field_sd=0.35,
)  # fmt: skip


@pytest.fixture
def game():
    return CalibrationGame(n_electrodes=64, params=CORRELATED, seed=0, resolution=64)


def _perfect_click(game):
    """Where an honest player would click: the flash's brightest pixel, or None if dark."""
    flash = game.flash()
    if flash.max() < 0.05:
        return None
    row, col = divmod(int(flash.argmax()), flash.shape[1])
    return row, col


def test_pixels_and_degrees_round_trip(game):
    for row, col in [(0, 0), (10, 50), (63, 63)]:
        x, y = game.pixel_to_deg(row, col)
        assert game.deg_to_pixel(x, y) == (row, col)


def test_a_visible_flash_appears_at_the_true_phosphene(game):
    # Force a strong flash on a working electrode and check where it lands.
    e = int(game.truth.alive.nonzero()[0])
    game.pending = (e, 100.0)
    row, col = _perfect_click(game)
    x, y = game.pixel_to_deg(row, col)
    assert abs(x - game.truth.x_deg[e].item()) < 1.0
    assert abs(y - game.truth.y_deg[e].item()) < 1.0


def test_a_flash_far_below_threshold_is_dark(game):
    e = int(game.truth.alive.nonzero()[0])
    game.pending = (e, 1.0)
    assert game.flash().max() < 0.05


def test_honest_answers_shrink_the_map_error(game):
    start = game.status()
    assert start["queries"] == 0 and start["reached_at"] is None
    assert start["exhaustive_queries"] == 64 * 8
    assert start["target_error_deg"] > 0
    for _ in range(40):
        game.answer(_perfect_click(game))
    end = game.status()
    assert end["queries"] == 40
    assert end["location_error_deg"] < 0.7 * start["location_error_deg"]


def test_autoplay_answers_for_the_user(game):
    game.autoplay(25)
    assert game.status()["queries"] == 25
    assert len(game.history) == 26  # the starting error plus one per answer


def test_map_image_is_an_rgb_picture(game):
    img = game.map_image()
    assert img.shape == (64, 64, 3) and img.dtype.name == "uint8"
    assert img.max() > 0
    assert game.map_image(size=300).shape == (300, 300, 3)
