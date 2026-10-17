"""Wedge-dipole retinotopic map between the visual field and the V1 surface.

Visual field positions are in degrees (x right, y up, fixation at the origin). Each
hemifield projects to the opposite hemisphere, so cortical positions carry a
``hemisphere`` label: +1 for the right visual field (left V1), -1 for the left visual field
(right V1). Within a hemisphere, cortical coordinates are in mm, with the foveal
representation at the origin and x growing with eccentricity.

The map is the wedge-dipole model of Polimeni et al. (2006) with the human parameters used
by dynaphos (van der Grinten et al., 2024): k and a are Horton & Hoyt's (1991) human
magnification constants. Polimeni's model was fitted to macaque V1, so this pairing is a
cross-species approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RetinotopyParams:
    k: float = 17.3  # mm; overall scale of V1
    a: float = 0.75  # deg; foveal singularity (Horton & Hoyt 1991)
    b: float = 120.0  # deg; peripheral singularity
    alpha: float = 0.95  # wedge compression of polar angle


DEFAULT_RETINOTOPY = RetinotopyParams()


def visual_field_to_cortex(
    x_deg: Tensor, y_deg: Tensor, params: RetinotopyParams = DEFAULT_RETINOTOPY
) -> tuple[Tensor, Tensor, Tensor]:
    """Map visual field points (deg) to (x_mm, y_mm, hemisphere) on V1.

    The left hemifield is mirrored onto the right before mapping, so both hemispheres share
    one cortical coordinate frame. Gradients are undefined exactly at fixation (0, 0).
    """
    hemisphere = torch.where(x_deg < 0, -1, 1)
    z = torch.complex(x_deg.abs(), y_deg)
    wedged = torch.polar(z.abs(), params.alpha * z.angle())
    w = params.k * torch.log(params.b * (wedged + params.a) / (params.a * (wedged + params.b)))
    return w.real, w.imag, hemisphere


def cortex_to_visual_field(
    x_mm: Tensor, y_mm: Tensor, hemisphere: Tensor, params: RetinotopyParams = DEFAULT_RETINOTOPY
) -> tuple[Tensor, Tensor]:
    """Inverse of :func:`visual_field_to_cortex`; returns visual field (x_deg, y_deg)."""
    w = torch.complex(x_mm, y_mm)
    # Solve w = k log(b (z + a) / (a (z + b))) for z.
    e = torch.exp(w / params.k) * (params.a / params.b)
    wedged = (params.b * e - params.a) / (1 - e)
    z = torch.polar(wedged.abs(), wedged.angle() / params.alpha)
    return z.real * hemisphere, z.imag


def magnification(
    eccentricity_deg: Tensor, params: RetinotopyParams = DEFAULT_RETINOTOPY
) -> Tensor:
    """Linear cortical magnification (mm of V1 per degree) along the horizontal meridian."""
    return params.k * (1 / (eccentricity_deg + params.a) - 1 / (eccentricity_deg + params.b))
