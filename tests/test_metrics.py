"""Tests for benchmark metrics (PROJECT_SPEC Phase 3)."""

import pytest
import torch

from phosphene_lab.eval.metrics import mean_ssim, stimulation_stats


def test_ssim_of_identical_images_is_one():
    x = torch.rand(3, 32, 32)
    assert mean_ssim(x, x) == pytest.approx(1.0)


def test_ssim_drops_for_unrelated_images():
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(3, 32, 32, generator=g), torch.rand(3, 32, 32, generator=g)
    assert mean_ssim(a, b) < 0.2


def test_ssim_prefers_a_blurred_copy_to_noise():
    target = torch.zeros(2, 32, 32)
    target[:, 8:24, 12:20] = 1.0
    blurred = torch.nn.functional.avg_pool2d(target[:, None], 5, 1, 2)[:, 0]
    noise = torch.rand(2, 32, 32, generator=torch.Generator().manual_seed(1))
    assert mean_ssim(blurred, target) > mean_ssim(noise, target)


def test_stimulation_stats():
    stim = torch.tensor([[0.0, 50.0, 30.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    stats = stimulation_stats(stim)
    assert stats["active_electrodes"] == pytest.approx(1.0)  # (2 + 0) / 2 frames
    assert stats["total_current_uA"] == pytest.approx(40.0)  # (80 + 0) / 2
