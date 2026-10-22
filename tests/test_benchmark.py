"""Tests for one benchmark cell: patient -> encode -> render -> observer -> metrics."""

import dataclasses

import pytest
import torch

from phosphene_lab.encoders.naive import NaiveEncoder
from phosphene_lab.eval.benchmark import (
    TaskData,
    evaluate_cell,
    make_patient,
    score_with_observer,
)
from phosphene_lab.simulator.phosphene import RenderConfig


def _bars(n, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    images = torch.zeros(n, 3, 16, 16)
    for i, label in enumerate(labels):
        images[i, :, 2:14, (2 if label == 0 else 10) : (6 if label == 0 else 14)] = 1.0
    return images, labels, images[:, :1].clone()


def _toy_data():
    train, test = _bars(96, 0), _bars(48, 1)
    return TaskData(*train, *test, n_classes=2)


def test_make_patient_applies_dropout():
    patient = make_patient(1024, dropout=0.3, variability="default", seed=0)
    dead = (~patient.true_phosphene_map().alive).float().mean().item()
    assert dead == pytest.approx(0.3, abs=0.05)


def test_variability_levels_order_threshold_spread():
    spread = {
        v: make_patient(1024, 0.0, v, seed=0).true_phosphene_map().threshold_uA.log().std()
        for v in ("low", "default", "high")
    }
    assert spread["low"] < spread["default"] < spread["high"]


def test_unknown_variability_is_an_error():
    with pytest.raises(ValueError, match="variability"):
        make_patient(64, 0.0, "extreme", seed=0)


def test_cell_reports_all_metrics_and_solves_an_easy_task():
    patient = make_patient(256, 0.0, "default", seed=0)
    result = evaluate_cell(
        _toy_data(), NaiveEncoder(), patient, RenderConfig(resolution=32), observer_epochs=6
    )
    assert set(result) == {"accuracy", "ssim", "active_electrodes", "total_current_uA"}
    assert result["accuracy"] > 0.9
    assert 0 < result["active_electrodes"] < 256


def test_cell_accepts_a_wrong_encoder_map():
    patient = make_patient(256, 0.0, "default", seed=0)
    truth = patient.true_phosphene_map()
    # A uniform shift moves the fitted image box too, so it changes nothing; scrambling
    # which electrode sits where is the kind of error that hurts.
    order = torch.randperm(len(truth), generator=torch.Generator().manual_seed(0))
    wrong = dataclasses.replace(truth, x_deg=truth.x_deg[order], y_deg=truth.y_deg[order])
    good = evaluate_cell(_toy_data(), NaiveEncoder(), patient, RenderConfig(resolution=32), 2)
    bad = evaluate_cell(
        _toy_data(), NaiveEncoder(), patient, RenderConfig(resolution=32), 2, encoder_map=wrong
    )
    assert bad["ssim"] < good["ssim"]


def test_fixed_observer_is_hurt_by_a_wrong_map_but_an_adaptive_one_is_not():
    patient = make_patient(256, 0.0, "default", seed=0)
    truth = patient.true_phosphene_map()
    # A left-right mirrored belief shows every image flipped: a consistent, learnable error.
    wrong = dataclasses.replace(truth, x_deg=-truth.x_deg)
    data, render = _toy_data(), RenderConfig(resolution=32)

    _, observer = evaluate_cell(
        data, NaiveEncoder(), patient, render, observer_epochs=6, return_observer=True
    )
    fixed_true = score_with_observer(observer, data, NaiveEncoder(), patient, render, truth)
    fixed_wrong = score_with_observer(observer, data, NaiveEncoder(), patient, render, wrong)
    adaptive_wrong = evaluate_cell(
        data, NaiveEncoder(), patient, render, observer_epochs=6, encoder_map=wrong
    )["accuracy"]
    assert fixed_true > 0.9
    assert fixed_wrong < 0.2  # the unadapted reader reads left bars as right bars
    assert adaptive_wrong > 0.9  # a reader trained on the flipped percepts recovers
