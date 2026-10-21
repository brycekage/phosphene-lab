"""Benchmark metrics: percept quality and stimulation cost.

* SSIM (structural similarity) between a percept and the E2 simplified target, both on the
  renderer's pixel grid. 1 = same structure; near 0 = unrelated.
* Active electrodes and total current per frame. Total current is a proxy for charge per
  frame, assuming every active electrode fires the same pulse train.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity
from torch import Tensor


def mean_ssim(percepts: Tensor, targets: Tensor) -> float:
    """Mean SSIM over a batch of (B, H, W) images in [0, 1]."""
    p = percepts.detach().cpu().numpy().astype(np.float64)
    t = targets.detach().cpu().numpy().astype(np.float64)
    scores = [structural_similarity(a, b, data_range=1.0) for a, b in zip(p, t, strict=True)]
    return float(np.mean(scores))


def stimulation_stats(stim_uA: Tensor) -> dict[str, float]:
    """Per-frame averages over a (B, N) batch of currents."""
    return {
        "active_electrodes": (stim_uA > 0).sum(dim=1).float().mean().item(),
        "total_current_uA": stim_uA.sum(dim=1).mean().item(),
    }
