"""Electrode array geometry on the V1 surface.

Electrodes live in per-hemisphere cortical coordinates (mm, see ``retinotopy``) inside a
rectangular cortical patch. Two layouts are supported:

* ``grid``: a regular lattice (Utah-array-like), split evenly across hemispheres.
* ``thread_insertion``: semi-random insertion sites with a minimum spacing, loosely modeled on
  flexible-thread arrays. Each site is one electrode; contacts along the thread's depth are
  not modeled. An optional exclusion mask (e.g. blood vessels from the insertion planner)
  forbids sites.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from phosphene_lab.simulator.retinotopy import (
    DEFAULT_RETINOTOPY,
    RetinotopyParams,
    cortex_to_visual_field,
)

# (x_mm, y_mm, hemisphere) -> bool tensor, True where an electrode must not be placed.
ExclusionMask = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class CorticalPatch:
    """Rectangle on each hemisphere's V1 map.

    The default spans roughly 1-8 deg eccentricity and maps entirely inside the correct
    hemifield and the 20 deg rendered field.
    """

    x_min_mm: float = 14.5
    x_max_mm: float = 41.4
    y_half_mm: float = 10.0

    @property
    def width_mm(self) -> float:
        return self.x_max_mm - self.x_min_mm

    @property
    def height_mm(self) -> float:
        return 2 * self.y_half_mm


DEFAULT_PATCH = CorticalPatch()


@dataclass(frozen=True)
class ElectrodeArray:
    cortex_x_mm: Tensor
    cortex_y_mm: Tensor
    hemisphere: Tensor  # +1: left V1 / right visual field, -1: right V1 / left visual field

    def __len__(self) -> int:
        return len(self.cortex_x_mm)

    def to_visual_field(
        self, params: RetinotopyParams = DEFAULT_RETINOTOPY
    ) -> tuple[Tensor, Tensor]:
        """Predicted phosphene centers (deg) from the retinotopic map alone."""
        return cortex_to_visual_field(self.cortex_x_mm, self.cortex_y_mm, self.hemisphere, params)

    @classmethod
    def grid(
        cls,
        n_electrodes: int,
        patch: CorticalPatch = DEFAULT_PATCH,
        hemispheres: Sequence[int] = (1, -1),
    ) -> ElectrodeArray:
        if n_electrodes % len(hemispheres):
            raise ValueError(f"{n_electrodes} electrodes cannot split evenly over {hemispheres}")
        per_hemisphere = n_electrodes // len(hemispheres)
        rows, cols = _closest_factorization(per_hemisphere, patch.width_mm / patch.height_mm)
        xs = torch.linspace(patch.x_min_mm, patch.x_max_mm, cols)
        ys = torch.linspace(-patch.y_half_mm, patch.y_half_mm, rows)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        return cls(
            cortex_x_mm=gx.flatten().repeat(len(hemispheres)),
            cortex_y_mm=gy.flatten().repeat(len(hemispheres)),
            hemisphere=torch.tensor(hemispheres).repeat_interleave(per_hemisphere),
        )

    @classmethod
    def thread_insertion(
        cls,
        n_electrodes: int,
        patch: CorticalPatch = DEFAULT_PATCH,
        hemispheres: Sequence[int] = (1, -1),
        min_spacing_mm: float = 0.4,
        exclusion_mask: ExclusionMask | None = None,
        seed: int = 0,
        max_candidates_per_site: int = 200,
    ) -> ElectrodeArray:
        """Sample insertion sites uniformly in the patch, rejecting excluded or crowded ones."""
        rng = np.random.default_rng(seed)
        counts = [n_electrodes // len(hemispheres)] * len(hemispheres)
        for i in range(n_electrodes % len(hemispheres)):
            counts[i] += 1

        xs, ys, hs = [], [], []
        for h, count in zip(hemispheres, counts, strict=True):
            x, y = _sample_sites(
                rng, h, count, patch, min_spacing_mm, exclusion_mask, max_candidates_per_site
            )
            xs.append(x)
            ys.append(y)
            hs.append(np.full(count, h))
        dtype = torch.get_default_dtype()
        return cls(
            cortex_x_mm=torch.from_numpy(np.concatenate(xs)).to(dtype),
            cortex_y_mm=torch.from_numpy(np.concatenate(ys)).to(dtype),
            hemisphere=torch.from_numpy(np.concatenate(hs)),
        )


def _closest_factorization(n: int, aspect: float) -> tuple[int, int]:
    """(rows, cols) with rows * cols == n and cols / rows closest to ``aspect``."""
    pairs = [(r, n // r) for r in range(1, n + 1) if n % r == 0]
    return min(pairs, key=lambda rc: abs(math.log(rc[1] / rc[0]) - math.log(aspect)))


def _sample_sites(
    rng: np.random.Generator,
    hemisphere: int,
    count: int,
    patch: CorticalPatch,
    min_spacing_mm: float,
    exclusion_mask: ExclusionMask | None,
    max_candidates_per_site: int,
) -> tuple[np.ndarray, np.ndarray]:
    accepted = np.empty((count, 2))
    n_accepted = 0
    budget = max(count, 1) * max_candidates_per_site
    min_d2 = min_spacing_mm**2
    while n_accepted < count and budget > 0:
        batch = min(1024, budget)
        budget -= batch
        cand = np.column_stack(
            [
                rng.uniform(patch.x_min_mm, patch.x_max_mm, batch),
                rng.uniform(-patch.y_half_mm, patch.y_half_mm, batch),
            ]
        )
        if exclusion_mask is not None:
            t = torch.from_numpy(cand)
            blocked = exclusion_mask(t[:, 0], t[:, 1], torch.full((batch,), hemisphere))
            cand = cand[~blocked.numpy().astype(bool)]
        for point in cand:
            existing = accepted[:n_accepted]
            if n_accepted and (((existing - point) ** 2).sum(axis=1) < min_d2).any():
                continue
            accepted[n_accepted] = point
            n_accepted += 1
            if n_accepted == count:
                break
    if n_accepted < count:
        raise RuntimeError(
            f"placed only {n_accepted}/{count} sites in hemisphere {hemisphere}; "
            "loosen min_spacing_mm, enlarge the patch, or shrink the exclusion mask"
        )
    return accepted[:, 0], accepted[:, 1]
