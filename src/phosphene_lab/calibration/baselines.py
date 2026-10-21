"""Baseline calibration procedures, each electrode treated on its own.

* ``SequentialCalibrator`` ("exhaustive"): electrode by electrode, a fixed protocol of
  bisection trials to find the threshold, then localization trials well above it. After
  every electrode has had its turn it starts another pass.
* ``RandomCalibrator``: the same per-electrode protocol, but each query goes to a random
  electrode, so partial budgets are spread over the whole array instead of the first rows.
"""

from __future__ import annotations

from phosphene_lab.calibration.common import Calibrator


class SequentialCalibrator(Calibrator):
    def next_query(self) -> tuple[int, float]:
        electrode = (self.queries // self.cfg.queries_per_electrode) % self.n
        return electrode, self.protocol_current(electrode)


class RandomCalibrator(Calibrator):
    def next_query(self) -> tuple[int, float]:
        electrode = self.rng.randrange(self.n)
        return electrode, self.protocol_current(electrode)
