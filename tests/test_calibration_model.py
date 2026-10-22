"""Tests for the calibration belief models: threshold grids and the cortical location GP."""

import pytest
import torch

from phosphene_lab.calibration.model import LocationModel, ThresholdGrid
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.retinotopy import cortex_to_visual_field

PARAMS = PatientParams()


def _grid(n=4):
    return ThresholdGrid(n, PARAMS, max_current_uA=100.0)


# --- ThresholdGrid --------------------------------------------------------------------


def test_threshold_posterior_converges_on_the_true_threshold():
    patient = VirtualPatient(ElectrodeArray.grid(64), PARAMS, seed=0)
    e = int(torch.nonzero(patient.true_phosphene_map().alive)[0])
    truth = patient.true_phosphene_map().threshold_uA[e].item()
    grid = ThresholdGrid(64, PARAMS, 100.0)
    for _ in range(40):
        current = grid.median(e)
        grid.update(e, current, patient.query_detect(e, current))
    assert grid.mean()[e].item() == pytest.approx(truth, abs=3.0)
    assert grid.p_dead()[e] < 0.05


def test_silence_at_full_current_means_dead():
    grid = _grid()
    for _ in range(6):
        grid.update(2, 100.0, False)
    assert grid.p_dead()[2] > 0.9
    assert grid.p_dead()[0] == pytest.approx(PARAMS.dropout_rate, abs=1e-6)


def test_information_peaks_near_the_believed_threshold():
    grid = _grid()
    for current, seen in [(10, False), (40, True), (25, True), (17, False), (21, True)]:
        grid.update(0, current, seen)
    currents = torch.tensor([5.0, 20.0, 90.0])
    info = grid.information(0, currents)
    assert info[1] > info[0] and info[1] > info[2]


def test_prior_can_be_replaced_without_losing_evidence():
    grid = _grid()
    for _ in range(10):
        grid.update(1, 60.0, True)  # strong evidence the threshold is below 60
    grid.set_prior(torch.tensor([1]), log_mean=torch.tensor([torch.log(torch.tensor(80.0))]),
                   log_sd=torch.tensor([0.1]))  # fmt: skip
    assert grid.mean()[1] < 65  # evidence still counts against the new prior


def test_vectorized_scores_match_per_electrode_ones():
    grid = _grid()
    grid.update(1, 30.0, True)
    grid.update(2, 30.0, False)
    currents = torch.tensor([10.0, 30.0, 70.0])
    info, seen = grid.information_all(currents), grid.p_seen_all(currents)
    for e in range(4):
        torch.testing.assert_close(info[e], grid.information(e, currents))
        torch.testing.assert_close(seen[e], grid.p_seen(e, currents))


def test_likelihood_only_estimate_ignores_the_prior():
    grid = _grid()
    for current, seen in [(30, True), (20, False), (25, True), (22, False), (24, True)]:
        grid.update(0, current, seen)
    before, _ = grid.likelihood_log_stats()
    grid.set_prior(torch.tensor([0]), torch.tensor([4.4]), torch.tensor([0.05]))  # ~80 uA
    after, sd = grid.likelihood_log_stats()
    assert after[0] == pytest.approx(before[0].item())
    # Lapses leave a tail toward low thresholds after only five answers, hence the loose SD.
    assert 20 < after[0].exp() < 26 and sd[0] < 0.6


def test_p_seen_is_a_valid_probability():
    p = _grid().p_seen(0, torch.tensor([0.0, 20.0, 100.0]))
    assert torch.all((p >= 0) & (p <= 1))
    assert p[0] < p[1] < p[2]


# --- LocationModel --------------------------------------------------------------------


def _model(array, share):
    return LocationModel(array, PARAMS, share=share, offset_sd_mm=1.5, smooth_sd_mm=0.5,
                         length_mm=5.0, jitter_sd_mm=0.3)  # fmt: skip


def _report(array, electrodes, shift_mm):
    """Noise-free reports of electrodes displaced by a common cortical shift."""
    x, y = cortex_to_visual_field(
        array.cortex_x_mm[electrodes] + shift_mm[0],
        array.cortex_y_mm[electrodes] + shift_mm[1],
        array.hemisphere[electrodes],
    )
    return x, y


def test_shared_model_spreads_a_common_offset_to_unvisited_electrodes():
    array = ElectrodeArray.grid(256)
    right = torch.nonzero(array.hemisphere == 1)[:, 0]
    visited, unvisited = right[::4], right[1::4]
    shared, independent = _model(array, True), _model(array, False)
    xs, ys = _report(array, visited, (1.0, -0.5))
    for model in (shared, independent):
        for e, x, y in zip(visited.tolist(), xs.tolist(), ys.tolist(), strict=True):
            model.add_report(e, x, y)
    dx_shared, dy_shared = shared.residual_mm()
    dx_indep, _ = independent.residual_mm()
    assert dx_shared[unvisited].mean() == pytest.approx(1.0, abs=0.2)
    assert dy_shared[unvisited].mean() == pytest.approx(-0.5, abs=0.2)
    assert dx_indep[unvisited].abs().max() < 1e-6  # nothing learned without sharing


def test_reports_shrink_posterior_variance():
    array = ElectrodeArray.grid(64)
    model = _model(array, True)
    before = model.variance_mm2().clone()
    x, y = _report(array, torch.tensor([5]), (0.0, 0.0))
    model.add_report(5, x.item(), y.item())
    after = model.variance_mm2()
    assert after[5] < before[5]
    assert torch.all(after <= before + 1e-9)


def _wrong_prior(array, offset_sd_mm, jitter_sd_mm):
    return LocationModel(array, PARAMS, share=True, offset_sd_mm=offset_sd_mm,
                         smooth_sd_mm=0.05, length_mm=5.0, jitter_sd_mm=jitter_sd_mm)  # fmt: skip


def _add_displaced(model, array, electrodes, dx, dy):
    x, y = cortex_to_visual_field(
        array.cortex_x_mm[electrodes] + dx, array.cortex_y_mm[electrodes] + dy,
        array.hemisphere[electrodes],
    )  # fmt: skip
    for e, xi, yi in zip(electrodes.tolist(), x.tolist(), y.tolist(), strict=True):
        model.add_report(e, xi, yi)


def test_fitting_recovers_a_large_per_electrode_jitter():
    array = ElectrodeArray.grid(256)
    g = torch.Generator().manual_seed(0)
    seen = torch.arange(0, 256, 2)
    model = _wrong_prior(array, offset_sd_mm=0.05, jitter_sd_mm=0.05)
    dx, dy = torch.randn(128, generator=g), torch.randn(128, generator=g)
    noise_sd = model.noise_var_mm2[seen].sqrt()  # pointing noise (~2 mm), as the model assumes
    for _ in range(8):  # one exhaustive pass of reports per electrode
        _add_displaced(
            model, array, seen, dx + noise_sd * torch.randn(128, generator=g),
            dy + noise_sd * torch.randn(128, generator=g),
        )  # fmt: skip
    fitted = model.fit_hyperparameters()
    assert 0.7 < fitted["jitter_sd_mm"] < 1.4  # truth 1.0, prior said 0.05


def test_fitting_lets_a_common_offset_reach_unvisited_electrodes():
    array = ElectrodeArray.grid(256)
    right = torch.nonzero(array.hemisphere == 1)[:, 0]
    visited, unvisited = right[::4], right[1::4]
    model = _wrong_prior(array, offset_sd_mm=0.05, jitter_sd_mm=1.0)  # prior: no shared shift
    _add_displaced(model, array, visited, 1.5, -1.0)
    before = model.residual_mm()[0][unvisited].mean().item()
    model.fit_hyperparameters()
    dx, dy = model.residual_mm()
    assert before < 0.3  # the wrong prior keeps unvisited electrodes near the plan
    assert dx[unvisited].mean() == pytest.approx(1.5, abs=0.3)
    assert dy[unvisited].mean() == pytest.approx(-1.0, abs=0.3)


def test_fitting_needs_enough_electrodes():
    array = ElectrodeArray.grid(64)
    model = _wrong_prior(array, offset_sd_mm=0.05, jitter_sd_mm=0.05)
    _add_displaced(model, array, torch.arange(5), 2.0, 0.0)
    k_before = model.K.clone()
    assert model.fit_hyperparameters(min_electrodes=10) is None
    assert torch.equal(model.K, k_before)


def test_estimated_map_is_in_degrees_and_matches_reports():
    array = ElectrodeArray.grid(64)
    model = _model(array, False)
    x, y = _report(array, torch.tensor([3]), (0.8, 0.0))
    for _ in range(20):  # many identical reports: posterior goes to the reported spot
        model.add_report(3, x.item(), y.item())
    ex, ey = model.locations_deg()
    assert ex[3].item() == pytest.approx(x.item(), abs=0.05)
    assert ey[3].item() == pytest.approx(y.item(), abs=0.05)
