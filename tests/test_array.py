"""Tests for electrode array geometry on the V1 surface."""

import pytest
import torch

from phosphene_lab.simulator.array import CorticalPatch, ElectrodeArray

SIZES = [64, 256, 1024, 1600]


@pytest.mark.parametrize("n", SIZES)
def test_grid_has_requested_count_split_across_hemispheres(n):
    arr = ElectrodeArray.grid(n)
    assert len(arr) == n
    assert (arr.hemisphere == 1).sum().item() == n // 2
    assert (arr.hemisphere == -1).sum().item() == n // 2


@pytest.mark.parametrize("make", ["grid", "thread"])
def test_phosphenes_land_in_rendered_field_on_correct_side(make):
    arr = ElectrodeArray.grid(1024) if make == "grid" else ElectrodeArray.thread_insertion(1024)
    x, y = arr.to_visual_field()
    assert (x.abs() <= 10).all() and (y.abs() <= 10).all()
    assert (torch.sign(x) == arr.hemisphere).all()


def test_grid_is_regular_within_each_hemisphere():
    arr = ElectrodeArray.grid(256)
    xs = torch.unique(arr.cortex_x_mm[arr.hemisphere == 1])
    ys = torch.unique(arr.cortex_y_mm[arr.hemisphere == 1])
    assert len(xs) * len(ys) == 128
    assert torch.allclose(torch.diff(xs), torch.diff(xs)[0].expand(len(xs) - 1))
    assert torch.allclose(torch.diff(ys), torch.diff(ys)[0].expand(len(ys) - 1))


def test_grid_rejects_odd_count_across_two_hemispheres():
    with pytest.raises(ValueError):
        ElectrodeArray.grid(63)


def test_single_hemisphere_grid():
    arr = ElectrodeArray.grid(64, hemispheres=(1,))
    assert (arr.hemisphere == 1).all() and len(arr) == 64


def test_thread_insertion_is_reproducible_per_seed():
    a = ElectrodeArray.thread_insertion(256, seed=1)
    b = ElectrodeArray.thread_insertion(256, seed=1)
    c = ElectrodeArray.thread_insertion(256, seed=2)
    assert torch.equal(a.cortex_x_mm, b.cortex_x_mm)
    assert not torch.equal(a.cortex_x_mm, c.cortex_x_mm)


def test_thread_insertion_respects_min_spacing():
    spacing = 0.5
    arr = ElectrodeArray.thread_insertion(1600, min_spacing_mm=spacing, seed=0)
    for h in (1, -1):
        m = arr.hemisphere == h
        pts = torch.stack([arr.cortex_x_mm[m], arr.cortex_y_mm[m]], dim=1)
        d = torch.cdist(pts, pts) + torch.eye(len(pts)) * 1e9
        assert d.min().item() >= spacing - 1e-9


def test_thread_insertion_avoids_exclusion_mask():
    patch = CorticalPatch()
    cx = (patch.x_min_mm + patch.x_max_mm) / 2

    def vessel(x_mm, y_mm, hemisphere):
        return (x_mm - cx).abs() < 3.0  # a 6 mm wide vertical "vessel" band

    arr = ElectrodeArray.thread_insertion(512, exclusion_mask=vessel, seed=0)
    assert not vessel(arr.cortex_x_mm, arr.cortex_y_mm, arr.hemisphere).any()
    assert len(arr) == 512


def test_thread_insertion_raises_when_patch_cannot_fit_sites():
    def everything(x_mm, y_mm, hemisphere):
        return torch.ones_like(x_mm, dtype=torch.bool)

    with pytest.raises(RuntimeError):
        ElectrodeArray.thread_insertion(64, exclusion_mask=everything, seed=0)
