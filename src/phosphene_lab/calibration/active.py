"""Bayesian active calibration: ask the question that teaches the most, and share answers.

Two ideas cut the number of queries:

1. **Sharing across electrodes.** The location model is a Gaussian process over cortex whose
   prior contains a common shift and a smooth deformation per array (placement error), so
   pointing reports from a few electrodes also move every neighbor's estimate. Threshold
   evidence is shared the same way: every few queries, a GP over cortex on the per-electrode
   (log) threshold estimates sets each electrode's prior from its neighbors, using
   leave-one-out predictions so no answer is counted twice.
2. **Choosing by expected information gain.** For every electrode and candidate current, it
   scores how much one answer is expected to reveal about the threshold (and dead/alive
   state), plus, weighted by the chance the flash is seen, how much a pointing report would
   reveal about the location. It asks the highest-scoring question.

The approach follows Bayesian active learning for psychophysics and prosthesis fitting; the
GP sharing is the "one query informs neighboring electrodes" idea from PROJECT_SPEC Phase 4.
"""

from __future__ import annotations

import math

import torch

from phosphene_lab.calibration.common import Calibrator


class ActiveCalibrator(Calibrator):
    share = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg, pop = self.cfg, self.cfg.population
        self.candidates = torch.arange(
            cfg.candidate_step_uA, cfg.max_current_uA + 1e-6, cfg.candidate_step_uA
        )
        # Prior covariance of log thresholds over cortex: smooth field + per-electrode part.
        pos = torch.stack([self.array.cortex_x_mm, self.array.cortex_y_mm], dim=1)
        same = (self.array.hemisphere[:, None] == self.array.hemisphere[None]).float()
        rbf = torch.exp(-(torch.cdist(pos, pos) ** 2) / (2 * pop.threshold_field_length_mm**2))
        self._log_k = same * pop.threshold_field_sd**2 * rbf
        self._log_k += max(pop.threshold_log_sd, 1e-3) ** 2 * torch.eye(self.n)
        self._log_mean = math.log(pop.threshold_median_uA)

    def next_query(self) -> tuple[int, float]:
        threshold_info = self.thresholds.information_all(self.candidates)  # (N, K)
        p_seen = self.thresholds.p_seen_all(self.candidates)
        location_info = self.locations.information()[:, None]  # (N, 1)
        score = threshold_info + p_seen * location_info
        best = int(score.argmax())
        electrode, k = divmod(best, len(self.candidates))
        return electrode, self.candidates[k].item()

    def after_query(self) -> None:
        if self.queries % self.cfg.prior_refresh_every == 0:
            self.refresh_threshold_priors()

    def refresh_threshold_priors(self) -> None:
        """Set every electrode's threshold prior from its neighbors' evidence (GP over
        cortex). Electrodes with evidence get leave-one-out predictions."""
        mean, sd = self.thresholds.likelihood_log_stats()
        observed = torch.nonzero(
            (self.thresholds.n_trials >= 2) & (self.thresholds.p_dead() < 0.5)
        )[:, 0]
        if len(observed) < 2:
            return
        y = mean[observed] - self._log_mean
        noise = sd[observed] ** 2
        k_oo = self._log_k[observed][:, observed] + torch.diag(noise)
        k_inv = torch.linalg.inv(k_oo)
        alpha = k_inv @ y
        k_all = self._log_k[:, observed]
        prior_mean = self._log_mean + k_all @ alpha
        prior_var = self._log_k.diagonal() - ((k_all @ k_inv) * k_all).sum(dim=1)
        # Leave-one-out for the observed electrodes (closed form for Gaussian processes).
        diag = k_inv.diagonal()
        loo_mean = self._log_mean + y - alpha / diag
        loo_var = 1 / diag - noise
        prior_mean[observed] = loo_mean
        prior_var[observed] = loo_var
        self.thresholds.set_prior(
            torch.arange(self.n), prior_mean, prior_var.clamp(min=0.05**2).sqrt()
        )
