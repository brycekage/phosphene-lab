"""Calibration game: the user plays the patient.

A hidden virtual patient has a true phosphene map. Each round, active calibration picks an
electrode and a current; the flash is rendered with the true map, and the user clicks where
they saw it (or says they saw nothing). Their answer updates the calibrator's beliefs exactly
as a real patient's would. The game tracks the map error after every answer and compares it
with what one full exhaustive pass (8 queries per electrode) reaches on the same patient.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from phosphene_lab.calibration.active import ActiveCalibrator
from phosphene_lab.calibration.baselines import SequentialCalibrator
from phosphene_lab.calibration.common import CalibrationConfig, map_errors
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneRenderer, RenderConfig

PATIENT_SEED_OFFSET = 900_000  # keeps game patients apart from the experiments' patients


class CalibrationGame:
    def __init__(
        self,
        n_electrodes: int = 64,
        params: PatientParams = PatientParams(),  # noqa: B008 - frozen dataclass
        seed: int = 0,
        resolution: int = 128,
    ) -> None:
        array = ElectrodeArray.grid(n_electrodes)
        self.patient = VirtualPatient(array, params, seed=PATIENT_SEED_OFFSET + seed)
        self.truth = self.patient.true_phosphene_map()
        cfg = CalibrationConfig.from_population(params)
        self.calibrator = ActiveCalibrator(array, cfg, seed)
        self.render = RenderConfig(resolution=resolution)
        self.renderer = PhospheneRenderer(self.truth, self.render)
        self.exhaustive_queries = n_electrodes * cfg.queries_per_electrode

        # The bar to beat: one full exhaustive pass, simulated on an identical patient.
        twin = VirtualPatient(array, params, seed=PATIENT_SEED_OFFSET + seed)
        [(_, exhaustive_map)] = SequentialCalibrator(array, cfg, seed).run(
            twin, [self.exhaustive_queries]
        )
        self.target_error_deg = self._error(exhaustive_map)

        self.history = [self._error(self.calibrator.estimate())]
        self.pending = self.calibrator.next_query()

    def _error(self, estimate) -> float:
        return map_errors(estimate, self.truth)["location_error_deg"]

    def pixel_to_deg(self, row: int, col: int) -> tuple[float, float]:
        half, px = self.render.field_deg / 2, self.render.field_deg / self.render.resolution
        return -half + (col + 0.5) * px, half - (row + 0.5) * px

    def deg_to_pixel(self, x_deg: float, y_deg: float) -> tuple[int, int]:
        return self.renderer.pixel_index(x_deg, y_deg)

    @torch.no_grad()
    def flash(self) -> np.ndarray:
        """What the patient sees for the pending query. (R, R) float32 in [0, 1]."""
        electrode, current = self.pending
        stim = torch.zeros(len(self.truth))
        stim[electrode] = current
        return self.renderer.render(stim).numpy().astype(np.float32)

    def _record(self, report: tuple[float, float] | None) -> None:
        electrode, current = self.pending
        self.calibrator.observe(electrode, current, report)
        self.calibrator.queries += 1
        self.calibrator.after_query()
        self.history.append(self._error(self.calibrator.estimate()))
        self.pending = self.calibrator.next_query()

    def answer(self, click: tuple[int, int] | None) -> None:
        """The user's answer: the (row, col) pixel they clicked, or None for "saw nothing"."""
        self._record(None if click is None else self.pixel_to_deg(*click))

    def autoplay(self, n: int) -> None:
        """Let the simulated patient answer the next ``n`` queries."""
        for _ in range(n):
            self._record(self.patient.query_location(*self.pending))

    def status(self) -> dict:
        queries = len(self.history) - 1
        reached = [q for q, err in enumerate(self.history) if err <= self.target_error_deg]
        reached_at = reached[0] if reached else None
        return {
            "queries": queries,
            "location_error_deg": self.history[-1],
            "target_error_deg": self.target_error_deg,
            "exhaustive_queries": self.exhaustive_queries,
            "reached_at": reached_at,
            "saved": None if reached_at is None else self.exhaustive_queries - reached_at,
        }

    def map_image(self, size: int | None = None) -> np.ndarray:
        """True phosphene positions (gray rings), the current estimate (green dots), and a
        red line joining each pair, which shrinks as calibration improves. (S, S, 3) uint8."""
        size = size or self.render.resolution
        half = self.render.field_deg / 2
        scale = size / self.render.field_deg

        def to_px(x: float, y: float) -> tuple[float, float]:
            return (x + half) * scale, (half - y) * scale

        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)
        estimate = self.calibrator.estimate()
        r = max(1.0, size / 160)
        for e in torch.nonzero(self.truth.alive)[:, 0].tolist():
            tx, ty = to_px(self.truth.x_deg[e].item(), self.truth.y_deg[e].item())
            ex, ey = to_px(estimate.x_deg[e].item(), estimate.y_deg[e].item())
            draw.line([(tx, ty), (ex, ey)], fill=(200, 60, 60), width=1)
            draw.ellipse([tx - r - 1, ty - r - 1, tx + r + 1, ty + r + 1], outline=(150, 150, 150))
            draw.ellipse([ex - r, ey - r, ex + r, ey + r], fill=(60, 230, 120))
        return np.asarray(img)
