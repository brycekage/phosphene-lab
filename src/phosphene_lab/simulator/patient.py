"""VirtualPatient: hidden per-electrode phosphene parameters plus noisy query responses.

A real implant's phosphene map differs from the retinotopic prediction: electrodes sit at
slightly different cortical positions than planned, thresholds vary, phosphene sizes vary,
and some electrodes do not work. The VirtualPatient samples these hidden parameters once
(per seed) and only exposes them through the two clinical questions calibration can ask:

* ``query_detect``: "did you see a flash?" (yes/no, psychometric curve plus lapses)
* ``query_location``: "point to where you saw it" (true location plus pointing noise)

``true_phosphene_map`` is the ground truth, for evaluation only; calibration code must not
read it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.phosphene import PhospheneMap
from phosphene_lab.simulator.retinotopy import (
    DEFAULT_RETINOTOPY,
    RetinotopyParams,
    cortex_to_visual_field,
)


@dataclass(frozen=True)
class PatientParams:
    location_jitter_mm: float = 1.0  # SD of true cortical position vs the planned one
    threshold_median_uA: float = 20.0
    threshold_log_sd: float = 0.4  # thresholds are log-normal
    size_scale_log_sd: float = 0.2  # per-electrode phosphene size multiplier, log-normal
    dropout_rate: float = 0.1  # fraction of electrodes that produce no phosphene
    pointing_noise_deg: float = 0.5  # SD of each coordinate of a pointing response
    lapse_rate: float = 0.05  # probability that a response is a random guess
    psychometric_slope_uA: float = 3.0  # SD of the cumulative-normal detection curve
    # Spatially correlated deviations (all off by default; Phase 4 turns them on). They are
    # drawn from a separate random stream so turning them on leaves the other draws unchanged.
    placement_offset_mm: float = 0.0  # SD of a rigid shift of each hemisphere's array
    placement_rotation_deg: float = 0.0  # SD of a rotation of each array about its center
    threshold_field_sd: float = 0.0  # SD of a smooth log-threshold field over cortex
    threshold_field_length_mm: float = 3.0  # correlation length of that field


DEFAULT_PATIENT = PatientParams()


class VirtualPatient:
    def __init__(
        self,
        array: ElectrodeArray,
        params: PatientParams = DEFAULT_PATIENT,
        seed: int = 0,
        retinotopy: RetinotopyParams = DEFAULT_RETINOTOPY,
    ) -> None:
        self.params = params
        self.n_electrodes = len(array)
        self.n_queries = 0

        g = torch.Generator().manual_seed(seed)
        g_corr = torch.Generator().manual_seed(seed + 2_000_003)
        n = self.n_electrodes
        cx, cy = _place(array, params, g_corr)
        jx = torch.randn(n, generator=g) * params.location_jitter_mm
        jy = torch.randn(n, generator=g) * params.location_jitter_mm
        cx, cy = cx + jx, cy + jy
        x_deg, y_deg = cortex_to_visual_field(cx, cy, array.hemisphere, retinotopy)
        log_field = _smooth_field(cx, cy, array.hemisphere, params, g_corr)
        thresholds = params.threshold_median_uA * torch.exp(
            params.threshold_log_sd * torch.randn(n, generator=g) + log_field
        )
        size_scale = torch.exp(params.size_scale_log_sd * torch.randn(n, generator=g))
        alive = torch.rand(n, generator=g) >= params.dropout_rate
        self._truth = PhospheneMap(
            x_deg=x_deg,
            y_deg=y_deg,
            threshold_uA=thresholds,
            size_scale=size_scale,
            alive=alive,
        )
        # Separate stream so the number of queries never changes the hidden parameters.
        self._response_rng = torch.Generator().manual_seed(seed + 1_000_003)

    def true_phosphene_map(self) -> PhospheneMap:
        """Ground truth for evaluation. Calibration methods must not call this."""
        return self._truth

    def query_detect(self, electrode: int, current_uA: float) -> bool:
        """Stimulate one electrode and ask whether the patient saw a flash."""
        self.n_queries += 1
        return self._sees(electrode, current_uA)

    def query_location(self, electrode: int, current_uA: float) -> tuple[float, float] | None:
        """Stimulate one electrode and ask where the flash appeared; None if nothing was seen."""
        self.n_queries += 1
        if not self._sees(electrode, current_uA):
            return None
        noise = torch.randn(2, generator=self._response_rng) * self.params.pointing_noise_deg
        x = self._truth.x_deg[electrode] + noise[0]
        y = self._truth.y_deg[electrode] + noise[1]
        return x.item(), y.item()

    def detection_probability(self, electrode: int, current_uA: float) -> float:
        """P(yes) = lapse / 2 + (1 - lapse) * Phi((I - threshold) / slope); Phi = 0 if dead."""
        p = self.params
        if self._truth.alive[electrode]:
            z = (current_uA - self._truth.threshold_uA[electrode]) / p.psychometric_slope_uA
            seen = torch.special.ndtr(torch.as_tensor(z)).item()
        else:
            seen = 0.0
        return p.lapse_rate / 2 + (1 - p.lapse_rate) * seen

    def _sees(self, electrode: int, current_uA: float) -> bool:
        u = torch.rand(1, generator=self._response_rng).item()
        return u < self.detection_probability(electrode, current_uA)


def _place(
    array: ElectrodeArray, params: PatientParams, g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Planned cortical positions after a rigid placement error (shift + rotation) per array."""
    x, y = array.cortex_x_mm.clone(), array.cortex_y_mm.clone()
    for h in (1, -1):
        side = array.hemisphere == h
        shift = torch.randn(2, generator=g) * params.placement_offset_mm
        angle = torch.deg2rad(torch.randn(1, generator=g) * params.placement_rotation_deg)
        if not side.any():
            continue
        x0, y0 = x[side].mean(), y[side].mean()
        dx, dy = x[side] - x0, y[side] - y0
        cos, sin = torch.cos(angle), torch.sin(angle)
        x[side] = x0 + cos * dx - sin * dy + shift[0]
        y[side] = y0 + sin * dx + cos * dy + shift[1]
    return x, y


def _smooth_field(
    x: torch.Tensor, y: torch.Tensor, hemisphere: torch.Tensor, params: PatientParams,
    g: torch.Generator, n_features: int = 256,
) -> torch.Tensor:  # fmt: skip
    """A smooth random field over cortex (squared-exponential covariance, via random
    Fourier features), drawn independently for each hemisphere. Zero when the SD is 0."""
    field = torch.zeros_like(x)
    for h in (1, -1):
        w = torch.randn(n_features, 2, generator=g) / params.threshold_field_length_mm
        b = torch.rand(n_features, generator=g) * 2 * torch.pi
        a = torch.randn(n_features, generator=g)
        side = hemisphere == h
        pos = torch.stack([x[side], y[side]], dim=1)
        features = torch.cos(pos @ w.T + b) * (2 / n_features) ** 0.5
        field[side] = params.threshold_field_sd * (features @ a)
    return field
