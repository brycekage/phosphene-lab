"""Tests for calibration procedures: exhaustive, random-order, and Bayesian active."""

import dataclasses

import pytest
import torch

from phosphene_lab.calibration.active import ActiveCalibrator
from phosphene_lab.calibration.baselines import RandomCalibrator, SequentialCalibrator
from phosphene_lab.calibration.common import CalibrationConfig, map_errors
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap

CORRELATED = PatientParams(
    location_jitter_mm=0.3, placement_offset_mm=1.5, placement_rotation_deg=5.0,
    threshold_log_sd=0.2, threshold_field_sd=0.35,
)  # fmt: skip
CFG = CalibrationConfig.from_population(CORRELATED)
METHODS = (SequentialCalibrator, RandomCalibrator, ActiveCalibrator)


def _patient(n=64, seed=0, params=CORRELATED):
    return VirtualPatient(ElectrodeArray.grid(n), params, seed=seed)


def patient_array(patient):
    return ElectrodeArray.grid(patient.n_electrodes)


class _Guarded(VirtualPatient):
    """A patient whose ground truth cannot be read while calibration runs."""

    def true_phosphene_map(self):
        raise AssertionError("calibration read the hidden ground truth")


@pytest.mark.parametrize("method", METHODS)
def test_calibrators_use_exactly_the_query_budget_and_never_peek(method):
    array = ElectrodeArray.grid(64)
    patient = _Guarded(array, CORRELATED, seed=1)
    snapshots = method(array, CFG, seed=0).run(patient, checkpoints=[0, 40, 90])
    assert patient.n_queries == 90
    assert [q for q, _ in snapshots] == [0, 40, 90]
    assert all(isinstance(m, PhospheneMap) and len(m) == 64 for _, m in snapshots)


@pytest.mark.parametrize("method", METHODS)
def test_errors_fall_as_queries_accumulate(method):
    patient = _patient()
    truth = patient.true_phosphene_map()
    snapshots = method(patient_array(patient), CFG, seed=0).run(patient, checkpoints=[0, 400])
    before, after = (map_errors(m, truth) for _, m in snapshots)
    assert after["threshold_error_uA"] < before["threshold_error_uA"]
    assert after["location_error_deg"] < before["location_error_deg"]


def test_sequential_visits_electrodes_in_order():
    array = ElectrodeArray.grid(64)
    calibrator = SequentialCalibrator(array, CFG, seed=0)
    calibrator.run(_patient(), checkpoints=[3 * CFG.queries_per_electrode])
    trials = calibrator.thresholds.n_trials
    assert torch.all(trials[:3] == CFG.queries_per_electrode)
    assert torch.all(trials[3:] == 0)


def test_active_beats_random_order_when_errors_are_correlated():
    results = {}
    for method in (RandomCalibrator, ActiveCalibrator):
        errs = []
        for seed in range(2):
            patient = _patient(n=64, seed=seed)
            truth = patient.true_phosphene_map()
            snapshots = method(patient_array(patient), CFG, seed=seed).run(patient, [120])
            est = snapshots[0][1]
            errs.append(map_errors(est, truth)["location_error_deg"])
        results[method.__name__] = sum(errs) / len(errs)
    assert results["ActiveCalibrator"] < results["RandomCalibrator"]


@pytest.mark.slow
def test_refitting_the_location_prior_rescues_a_wrong_population_guess():
    # The calibrator assumes the Phase 1-3 population (no shared placement error); the
    # patient has a large shared offset. Refitting the GP hyperparameters should find it.
    wrong = CalibrationConfig.from_population(PatientParams())
    errs = {}
    for every in (0, 50):
        cfg = dataclasses.replace(wrong, fit_location_every=every)
        errs[every] = []
        for seed in range(2):
            patient = _patient(seed=seed)
            calibrator = ActiveCalibrator(patient_array(patient), cfg, seed=seed)
            ((_, estimate),) = calibrator.run(patient, checkpoints=[200])
            errs[every].append(map_errors(estimate, patient.true_phosphene_map()))
    fixed = sum(e["location_error_deg"] for e in errs[0])
    fitted = sum(e["location_error_deg"] for e in errs[50])
    assert fitted < 0.8 * fixed


def test_map_errors_on_a_known_pair():
    truth = PhospheneMap(
        x_deg=torch.tensor([1.0, 2.0, 3.0]),
        y_deg=torch.zeros(3),
        threshold_uA=torch.tensor([20.0, 30.0, 40.0]),
        alive=torch.tensor([True, True, False]),
    )
    estimate = dataclasses.replace(
        truth,
        x_deg=torch.tensor([1.5, 2.0, 9.0]),
        threshold_uA=torch.tensor([25.0, 30.0, 0.0]),
        alive=torch.tensor([True, False, False]),
    )
    errors = map_errors(estimate, truth)
    assert errors["location_error_deg"] == pytest.approx(0.25)  # alive only: (0.5 + 0) / 2
    assert errors["threshold_error_uA"] == pytest.approx(2.5)
    assert errors["dead_accuracy"] == pytest.approx(2 / 3)
