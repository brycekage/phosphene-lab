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
        n = self.n_electrodes
        jx = torch.randn(n, generator=g) * params.location_jitter_mm
        jy = torch.randn(n, generator=g) * params.location_jitter_mm
        x_deg, y_deg = cortex_to_visual_field(
            array.cortex_x_mm + jx, array.cortex_y_mm + jy, array.hemisphere, retinotopy
        )
        thresholds = params.threshold_median_uA * torch.exp(
            params.threshold_log_sd * torch.randn(n, generator=g)
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
