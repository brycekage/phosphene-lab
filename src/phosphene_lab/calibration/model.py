"""What the calibration system believes about each electrode, updated query by query.

Two belief models, both built only from query responses and population-level knowledge
(the psychometric slope, lapse rate, pointing noise, and how much patients vary), never from
a patient's hidden parameters:

* ``ThresholdGrid``: for each electrode, a probability over thresholds on a 1-uA grid, plus a
  "dead" state. Each yes/no answer multiplies in the psychometric likelihood.
* ``LocationModel``: a Gaussian process over cortex for how far each electrode really sits
  from where it was planned. With ``share=True`` the prior includes a common offset per
  hemisphere and a smooth field (array placement error), so one electrode's report also moves
  its neighbors' estimates. With ``share=False`` every electrode is on its own.

Working in cortical millimeters rather than degrees matters: the same placement error is a
tiny shift near the fovea and a large one in the periphery, but it is one number on cortex.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams
from phosphene_lab.simulator.retinotopy import (
    DEFAULT_RETINOTOPY,
    RetinotopyParams,
    cortex_to_visual_field,
    magnification,
    visual_field_to_cortex,
)


class ThresholdGrid:
    def __init__(
        self, n: int, params: PatientParams, max_current_uA: float, step_uA: float = 1.0
    ) -> None:
        self.params = params
        self.levels = torch.arange(step_uA, max_current_uA + step_uA / 2, step_uA)
        self.log_lik = torch.zeros(n, len(self.levels) + 1)  # last column: dead
        self.n_trials = torch.zeros(n, dtype=torch.long)
        total_sd = math.hypot(params.threshold_log_sd, params.threshold_field_sd)
        self.log_prior = torch.empty_like(self.log_lik)
        self.set_prior(
            torch.arange(n),
            torch.full((n,), math.log(params.threshold_median_uA)),
            torch.full((n,), total_sd),
        )

    def set_prior(self, electrodes: Tensor, log_mean: Tensor, log_sd: Tensor) -> None:
        """Log-normal prior over alive thresholds (truncated to the grid) plus P(dead)."""
        z = (self.levels.log()[None] - log_mean[:, None]) / log_sd[:, None]
        alive = -0.5 * z**2 - self.levels.log()[None]  # log-normal density, up to a constant
        alive = alive - torch.logsumexp(alive, dim=1, keepdim=True)
        p_dead = self.params.dropout_rate
        self.log_prior[electrodes, :-1] = alive + math.log(1 - p_dead)
        self.log_prior[electrodes, -1] = math.log(p_dead)

    def _p_yes(self, current_uA: Tensor) -> Tensor:
        """P(yes | state) for each current. (K,) -> (K, levels + 1)."""
        lapse = self.params.lapse_rate
        z = (current_uA[:, None] - self.levels[None]) / self.params.psychometric_slope_uA
        alive = lapse / 2 + (1 - lapse) * torch.special.ndtr(z)
        dead = torch.full((len(current_uA), 1), lapse / 2)
        return torch.cat([alive, dead], dim=1)

    def update(self, electrode: int, current_uA: float, seen: bool) -> None:
        p = self._p_yes(torch.tensor([float(current_uA)]))[0]
        self.log_lik[electrode] += torch.log(p if seen else 1 - p)
        self.n_trials[electrode] += 1

    def posterior(self) -> Tensor:
        return torch.softmax(self.log_prior + self.log_lik, dim=1)

    def p_dead(self) -> Tensor:
        return self.posterior()[:, -1]

    def _alive_posterior(self) -> Tensor:
        alive = self.posterior()[:, :-1]
        return alive / alive.sum(dim=1, keepdim=True).clamp(min=1e-12)

    def mean(self) -> Tensor:
        """Posterior mean threshold (uA), assuming the electrode works."""
        return self._alive_posterior() @ self.levels

    def log_mean_sd(self) -> tuple[Tensor, Tensor]:
        """Posterior mean and SD of log threshold, assuming the electrode works."""
        post, logs = self._alive_posterior(), self.levels.log()
        mean = post @ logs
        var = post @ logs**2 - mean**2
        return mean, var.clamp(min=1e-6).sqrt()

    def median(self, electrode: int) -> float:
        cdf = self._alive_posterior()[electrode].cumsum(0)
        index = min(int(torch.searchsorted(cdf, torch.tensor(0.5))), len(self.levels) - 1)
        return self.levels[index].item()

    def p_seen(self, electrode: int, currents_uA: Tensor) -> Tensor:
        return self._p_yes(currents_uA) @ self.posterior()[electrode]

    def information(self, electrode: int, currents_uA: Tensor) -> Tensor:
        """Expected information (nats) one yes/no answer at each current gives about the
        electrode's state: H(answer) - E_state[H(answer | state)]."""
        p_state = self._p_yes(currents_uA)
        post = self.posterior()[electrode]
        p = p_state @ post
        return _binary_entropy(p) - _binary_entropy(p_state) @ post

    def p_seen_all(self, currents_uA: Tensor) -> Tensor:
        """P(yes) for every electrode at every current. Returns (N, K)."""
        return self.posterior() @ self._p_yes(currents_uA).T

    def information_all(self, currents_uA: Tensor) -> Tensor:
        """``information`` for every electrode at once. Returns (N, K)."""
        p_state = self._p_yes(currents_uA)
        post = self.posterior()
        return _binary_entropy(post @ p_state.T) - post @ _binary_entropy(p_state).T

    def likelihood_log_stats(self) -> tuple[Tensor, Tensor]:
        """Mean and SD of log threshold from the answers alone (flat prior in log current),
        so neighbors can share evidence without counting any prior twice."""
        logs = self.levels.log()
        post = torch.softmax(self.log_lik[:, :-1] - logs[None], dim=1)  # log-uniform prior
        mean = post @ logs
        var = post @ logs**2 - mean**2
        return mean, var.clamp(min=1e-6).sqrt()


_HYPER_NAMES = ("offset_sd_mm", "smooth_sd_mm", "length_mm", "jitter_sd_mm")
# Weak prior for fitting: SDs around 1 mm and the length scale around 5 mm, each within a
# factor of ~e^1.5 (one SD in log space), and every value kept in [0.01, 20] mm.
_PRIOR_LOG_MEAN = torch.tensor([0.0, 0.0, math.log(5.0), 0.0])
_PRIOR_LOG_SD = torch.tensor([1.5, 1.5, 1.0, 1.5])
_LOG_MIN, _LOG_MAX = math.log(0.01), math.log(20.0)


def _binary_entropy(p: Tensor) -> Tensor:
    p = p.clamp(1e-9, 1 - 1e-9)
    return -(p * p.log() + (1 - p) * (1 - p).log())


class LocationModel:
    def __init__(
        self,
        array: ElectrodeArray,
        params: PatientParams,
        share: bool,
        offset_sd_mm: float,
        smooth_sd_mm: float,
        length_mm: float,
        jitter_sd_mm: float,
        retinotopy: RetinotopyParams = DEFAULT_RETINOTOPY,
    ) -> None:
        self.x0, self.y0 = array.cortex_x_mm, array.cortex_y_mm
        self.hemisphere = array.hemisphere
        self.retinotopy = retinotopy
        self.share = share
        n = len(array)
        pos = torch.stack([self.x0, self.y0], dim=1)
        self._d2 = torch.cdist(pos, pos) ** 2
        self._same = (self.hemisphere[:, None] == self.hemisphere[None]).float()
        self.hyper = {
            "offset_sd_mm": offset_sd_mm, "smooth_sd_mm": smooth_sd_mm,
            "length_mm": length_mm, "jitter_sd_mm": jitter_sd_mm,
        }  # fmt: skip
        self.K = self._kernel()
        # A pointing error of s degrees is s * M millimeters of cortex at that eccentricity.
        x_deg, y_deg = cortex_to_visual_field(self.x0, self.y0, self.hemisphere, retinotopy)
        mag = magnification(torch.hypot(x_deg, y_deg), retinotopy)
        self.noise_var_mm2 = (params.pointing_noise_deg * mag) ** 2
        self.sum_dx = torch.zeros(n)
        self.sum_dy = torch.zeros(n)
        self.count = torch.zeros(n)
        self._cache: tuple[Tensor, Tensor, Tensor] | None = None

    def add_report(self, electrode: int, x_deg: float, y_deg: float) -> None:
        # A report just across the vertical meridian still belongs to this electrode's
        # hemisphere, so mirror it into the right hemifield before mapping to cortex.
        h = int(self.hemisphere[electrode])
        x = torch.tensor([abs(x_deg) * h])
        cx, cy, _ = visual_field_to_cortex(x, torch.tensor([y_deg]), self.retinotopy)
        self.sum_dx[electrode] += cx.item() - self.x0[electrode].item()
        self.sum_dy[electrode] += cy.item() - self.y0[electrode].item()
        self.count[electrode] += 1
        self._cache = None

    def _kernel(self) -> Tensor:
        c, s, length, j = (self.hyper[k] for k in _HYPER_NAMES)
        n = len(self.x0)
        if not self.share:
            return (c**2 + s**2 + j**2) * torch.eye(n)
        rbf = torch.exp(-self._d2 / (2 * length**2))
        return self._same * (c**2 + s**2 * rbf) + j**2 * torch.eye(n)

    def fit_hyperparameters(
        self, min_electrodes: int = 10, max_iter: int = 50
    ) -> dict[str, float] | None:
        """Re-estimate the shared-model prior (offset, smooth field, length scale, jitter)
        from this patient's own reports: maximize the GP marginal likelihood of the mean
        report per electrode, plus a weak log-normal prior, since the shared offset is only
        one draw per hemisphere. Needs reports from ``min_electrodes``; returns the new
        values, or None if it did nothing."""
        seen = torch.nonzero(self.count > 0)[:, 0]
        if not self.share or len(seen) < min_electrodes:
            return None
        y = torch.stack([self.sum_dx[seen], self.sum_dy[seen]], 1) / self.count[seen, None]
        noise = torch.diag(self.noise_var_mm2[seen] / self.count[seen])
        d2, same = self._d2[seen][:, seen], self._same[seen][:, seen]
        eye = torch.eye(len(seen))
        start = [max(self.hyper[k], 0.01) for k in _HYPER_NAMES]
        log_p = torch.tensor(start).log().requires_grad_()
        optimizer = torch.optim.LBFGS([log_p], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            optimizer.zero_grad()
            c, s, length, j = log_p.clamp(_LOG_MIN, _LOG_MAX).exp()
            k = same * (c**2 + s**2 * torch.exp(-d2 / (2 * length**2))) + j**2 * eye + noise
            chol = torch.linalg.cholesky(k)
            # Two coordinates, each with 0.5 y^T K^-1 y + 0.5 log det K.
            nll = 0.5 * (y * torch.cholesky_solve(y, chol)).sum()
            nll = nll + 2 * chol.diagonal().log().sum()
            nll = nll + 0.5 * (((log_p - _PRIOR_LOG_MEAN) / _PRIOR_LOG_SD) ** 2).sum()
            nll.backward()
            return nll

        optimizer.step(closure)
        fitted = log_p.detach().clamp(_LOG_MIN, _LOG_MAX).exp().tolist()
        self.hyper = dict(zip(_HYPER_NAMES, fitted, strict=True))
        self.K = self._kernel()
        self._cache = None
        return dict(self.hyper)

    def _posterior(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._cache is None:
            seen = torch.nonzero(self.count > 0)[:, 0]
            if len(seen) == 0:
                zeros = torch.zeros(len(self.count))
                self._cache = (zeros, zeros.clone(), self.K.diagonal().clone())
            else:
                # Each electrode's reports collapse to their mean, with noise variance / count.
                y = torch.stack([self.sum_dx[seen], self.sum_dy[seen]], 1) / self.count[seen, None]
                noise = self.noise_var_mm2[seen] / self.count[seen]
                k_all = self.K[:, seen]
                chol = torch.linalg.cholesky(self.K[seen][:, seen] + torch.diag(noise))
                mean = k_all @ torch.cholesky_solve(y, chol)
                v = torch.linalg.solve_triangular(chol, k_all.T, upper=False)
                var = self.K.diagonal() - (v**2).sum(dim=0)
                self._cache = (mean[:, 0], mean[:, 1], var.clamp(min=1e-9))
        return self._cache

    def residual_mm(self) -> tuple[Tensor, Tensor]:
        dx, dy, _ = self._posterior()
        return dx, dy

    def variance_mm2(self) -> Tensor:
        """Posterior variance of each coordinate of the residual (the same for x and y)."""
        return self._posterior()[2]

    def locations_deg(self) -> tuple[Tensor, Tensor]:
        dx, dy = self.residual_mm()
        return cortex_to_visual_field(self.x0 + dx, self.y0 + dy, self.hemisphere, self.retinotopy)

    def information(self) -> Tensor:
        """Expected information (nats) about each electrode's location from one more report:
        two coordinates, each 0.5 * log(1 + prior variance / noise variance)."""
        return torch.log1p(self.variance_mm2() / self.noise_var_mm2)
