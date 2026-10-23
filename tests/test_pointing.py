"""Tests for the pointing task: point at a dot the implant shows, measure the miss in degrees."""

import math

import pytest
import torch

from phosphene_lab.encoders.naive import NaiveEncoder
from phosphene_lab.eval.pointing import centroid_deg, dot_images, pointing_errors
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap, RenderConfig

RENDER = RenderConfig(resolution=32, field_deg=20.0)


def test_dot_images_draw_one_disc_at_each_center():
    images, centers = dot_images(5, size=32, radius_frac=0.1, seed=0)
    assert images.shape == (5, 3, 32, 32) and centers.shape == (5, 2)
    for image, (u, v) in zip(images, centers.tolist(), strict=True):
        row, col = int(v * 32), int(u * 32)
        assert image[0, row, col] == 1.0  # the center is lit
        assert 0 < image[0].mean() < 0.1  # and the disc is small


def test_centroid_of_a_single_pixel_is_its_center():
    field = torch.zeros(1, 32, 32)
    field[0, 0, 31] = 1.0  # top-right pixel
    x, y, seen = centroid_deg(field, RENDER)
    half_px = 20.0 / 32 / 2
    assert x.item() == pytest.approx(10.0 - half_px)
    assert y.item() == pytest.approx(10.0 - half_px)
    assert seen.item()
    assert not centroid_deg(torch.zeros(1, 32, 32), RENDER)[2].item()


def test_a_wrong_map_makes_the_patient_point_elsewhere():
    patient = VirtualPatient(ElectrodeArray.grid(256), seed=0)
    truth = patient.true_phosphene_map()
    mirrored = PhospheneMap(
        x_deg=-truth.x_deg, y_deg=truth.y_deg, threshold_uA=truth.threshold_uA, alive=truth.alive
    )
    images, _ = dot_images(40, size=48, radius_frac=0.1, seed=1)
    encoder = NaiveEncoder(config=RENDER)
    right = pointing_errors(encoder, images, truth, truth, RENDER)
    wrong = pointing_errors(encoder, images, mirrored, truth, RENDER)
    assert right.nanmean() < 2.0
    assert wrong.nanmean() > 3 * right.nanmean()
    assert not math.isnan(right.nanmean().item())
