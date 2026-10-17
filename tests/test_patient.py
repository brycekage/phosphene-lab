"""Tests for the VirtualPatient: hidden phosphene parameters and noisy query responses."""

import pytest
import torch

from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap

ARRAY = ElectrodeArray.grid(1600)


def _first_alive(patient):
    return int(torch.nonzero(patient.true_phosphene_map().alive)[0])


def _first_dead(patient):
    return int(torch.nonzero(~patient.true_phosphene_map().alive)[0])


def test_same_seed_same_patient_different_seed_different_patient():
    a = VirtualPatient(ARRAY, seed=7).true_phosphene_map()
    b = VirtualPatient(ARRAY, seed=7).true_phosphene_map()
    c = VirtualPatient(ARRAY, seed=8).true_phosphene_map()
    for field in ("x_deg", "y_deg", "threshold_uA", "size_scale", "alive"):
        assert torch.equal(getattr(a, field), getattr(b, field))
    assert not torch.equal(a.x_deg, c.x_deg)


def test_true_locations_deviate_from_retinotopic_prediction_by_jitter():
    predicted = PhospheneMap.from_array(ARRAY)
    exact = VirtualPatient(ARRAY, PatientParams(location_jitter_mm=0.0)).true_phosphene_map()
    assert torch.allclose(exact.x_deg, predicted.x_deg, atol=1e-5)
    jittered = VirtualPatient(ARRAY, PatientParams(location_jitter_mm=1.0)).true_phosphene_map()
    error = torch.hypot(jittered.x_deg - predicted.x_deg, jittered.y_deg - predicted.y_deg)
    assert error.median() > 0.05


def test_thresholds_are_lognormal_around_median():
    params = PatientParams(threshold_median_uA=20.0, threshold_log_sd=0.4)
    thr = VirtualPatient(ARRAY, params).true_phosphene_map().threshold_uA
    assert thr.min() > 0
    assert thr.median().item() == pytest.approx(20.0, rel=0.1)
    assert thr.log().std().item() == pytest.approx(0.4, rel=0.1)


def test_dropout_fraction_matches_rate():
    alive = VirtualPatient(ARRAY, PatientParams(dropout_rate=0.3)).true_phosphene_map().alive
    assert (~alive).float().mean().item() == pytest.approx(0.3, abs=0.03)


def test_pointing_responses_scatter_around_true_location():
    params = PatientParams(pointing_noise_deg=0.5, lapse_rate=0.0)
    patient = VirtualPatient(ARRAY, params, seed=1)
    e = _first_alive(patient)
    true = patient.true_phosphene_map()
    pts = torch.tensor([patient.query_location(e, current_uA=100.0) for _ in range(2000)])
    assert pts[:, 0].mean().item() == pytest.approx(true.x_deg[e].item(), abs=0.05)
    assert pts[:, 1].mean().item() == pytest.approx(true.y_deg[e].item(), abs=0.05)
    assert pts.std(dim=0).mean().item() == pytest.approx(0.5, rel=0.1)


def test_detection_probability_rises_with_current():
    params = PatientParams(lapse_rate=0.0, psychometric_slope_uA=3.0)
    patient = VirtualPatient(ARRAY, params, seed=2)
    e = _first_alive(patient)
    thr = patient.true_phosphene_map().threshold_uA[e].item()

    def rate(current):
        return sum(patient.query_detect(e, current) for _ in range(1000)) / 1000

    low, mid, high = rate(thr - 6), rate(thr), rate(thr + 6)
    assert low < mid < high
    assert mid == pytest.approx(0.5, abs=0.06)
    assert high > 0.95


def test_lapses_cap_detection_below_one():
    patient = VirtualPatient(ARRAY, PatientParams(lapse_rate=0.2), seed=3)
    e = _first_alive(patient)
    rate = sum(patient.query_detect(e, 100.0) for _ in range(2000)) / 2000
    assert rate == pytest.approx(0.9, abs=0.03)


def test_dead_electrodes_never_seen_without_lapses():
    patient = VirtualPatient(ARRAY, PatientParams(dropout_rate=0.3, lapse_rate=0.0), seed=4)
    e = _first_dead(patient)
    assert not any(patient.query_detect(e, 100.0) for _ in range(500))
    assert patient.query_location(e, current_uA=100.0) is None


def test_dead_electrodes_draw_false_alarms_at_half_lapse_rate():
    patient = VirtualPatient(ARRAY, PatientParams(dropout_rate=0.3, lapse_rate=0.2), seed=5)
    e = _first_dead(patient)
    rate = sum(patient.query_detect(e, 100.0) for _ in range(2000)) / 2000
    assert rate == pytest.approx(0.1, abs=0.03)


def test_query_counter_and_reproducible_responses():
    def run(seed):
        p = VirtualPatient(ARRAY, seed=seed)
        e = _first_alive(p)
        answers = [p.query_detect(e, 25.0) for _ in range(20)]
        answers.append(p.query_location(e, current_uA=100.0))
        return p.n_queries, answers

    n, a = run(9)
    assert n == 21
    assert run(9)[1] == a
