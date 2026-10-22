"""Shared calibration machinery: the query loop, the estimated map, and error metrics.

A calibrator repeatedly picks one electrode and one current, stimulates, and asks the patient
to point at the flash (``query_location``; "nothing seen" is itself an answer). Every answer
updates the threshold grid and, when something was seen, the location model. The methods
differ only in which (electrode, current) they ask about next, and in whether neighboring
electrodes share information.

Each query is assumed to take about 5 s of clinic time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from phosphene_lab.calibration.model import LocationModel, ThresholdGrid
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap

SECONDS_PER_QUERY = 5.0


@dataclass(frozen=True)
class CalibrationConfig:
    """Population-level knowledge plus procedure settings. Nothing patient-specific."""

    population: PatientParams
    offset_sd_mm: float
    smooth_sd_mm: float
    length_mm: float
    jitter_sd_mm: float
    max_current_uA: float = 100.0
    threshold_trials: int = 5  # baseline protocol: bisection trials before localizing
    queries_per_electrode: int = 8  # baseline protocol: total queries per electrode
    candidate_step_uA: float = 5.0  # active method: currents it may choose from
    prior_refresh_every: int = 10  # active method: share threshold evidence every k queries
    fit_location_every: int = 0  # active method: refit the location prior every k (0 = never)

    @classmethod
    def from_population(cls, params: PatientParams, array_radius_mm: float = 8.0, **kw):
        # A small rotation about the array center moves an electrode r * angle millimeters,
        # a smooth field over the array; a long RBF length scale stands in for it.
        rotation = math.radians(params.placement_rotation_deg) * array_radius_mm
        return cls(
            population=params,
            offset_sd_mm=params.placement_offset_mm,
            smooth_sd_mm=max(rotation, 1e-3),
            length_mm=10.0,
            jitter_sd_mm=max(params.location_jitter_mm, 1e-3),
            **kw,
        )


class Calibrator:
    share = False  # whether the location model pools information across electrodes

    def __init__(self, array: ElectrodeArray, cfg: CalibrationConfig, seed: int = 0) -> None:
        self.array, self.cfg, self.n = array, cfg, len(array)
        self.rng = random.Random(seed)
        self.thresholds = ThresholdGrid(self.n, cfg.population, cfg.max_current_uA)
        self.locations = LocationModel(
            array, cfg.population, self.share, cfg.offset_sd_mm, cfg.smooth_sd_mm,
            cfg.length_mm, cfg.jitter_sd_mm,
        )  # fmt: skip
        self.queries = 0

    def next_query(self) -> tuple[int, float]:
        raise NotImplementedError

    def after_query(self) -> None:
        """Hook for methods that update shared priors."""

    def observe(self, electrode: int, current_uA: float, report) -> None:
        self.thresholds.update(electrode, current_uA, report is not None)
        if report is not None:
            self.locations.add_report(electrode, *report)

    def protocol_current(self, electrode: int) -> float:
        """Baseline rule: bisect on the threshold first, then localize well above it."""
        median = self.thresholds.median(electrode)
        if self.thresholds.n_trials[electrode] < self.cfg.threshold_trials:
            return median
        return min(self.cfg.max_current_uA, 1.5 * median + 5.0)

    def estimate(self) -> PhospheneMap:
        x, y = self.locations.locations_deg()
        return PhospheneMap(
            x_deg=x, y_deg=y, threshold_uA=self.thresholds.mean(),
            alive=self.thresholds.p_dead() < 0.5,
        )  # fmt: skip

    def run(
        self, patient: VirtualPatient, checkpoints: list[int]
    ) -> list[tuple[int, PhospheneMap]]:
        """Query until each checkpoint (total queries) and snapshot the estimated map there."""
        snapshots = []
        for target in sorted(checkpoints):
            while self.queries < target:
                electrode, current = self.next_query()
                self.observe(electrode, current, patient.query_location(electrode, current))
                self.queries += 1
                self.after_query()
            snapshots.append((self.queries, self.estimate()))
        return snapshots


def map_errors(estimate: PhospheneMap, truth: PhospheneMap) -> dict[str, float]:
    """Location and threshold error over electrodes that really work, and how often the
    working/dead call is right."""
    alive = truth.alive
    location = torch.hypot(estimate.x_deg - truth.x_deg, estimate.y_deg - truth.y_deg)
    threshold = (estimate.threshold_uA - truth.threshold_uA).abs()
    return {
        "location_error_deg": location[alive].mean().item(),
        "threshold_error_uA": threshold[alive].mean().item(),
        "dead_accuracy": (estimate.alive == truth.alive).float().mean().item(),
    }
