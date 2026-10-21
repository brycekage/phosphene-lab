"""Tests for the virtual observer: a small classifier that reads percepts (PROJECT_SPEC Phase 3)."""

import torch

from phosphene_lab.eval.observer import Observer, accuracy, train_observer


def _toy_task(n=128, size=32, seed=0):
    """Class 0: bright left half. Class 1: bright right half. Plus noise."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (n,), generator=g)
    x = torch.rand(n, size, size, generator=g) * 0.3
    for i, label in enumerate(labels):
        cols = slice(0, size // 2) if label == 0 else slice(size // 2, size)
        x[i, :, cols] += 0.6
    return x, labels


def test_observer_outputs_one_logit_per_class():
    logits = Observer(n_classes=7)(torch.rand(5, 1, 32, 32))
    assert logits.shape == (5, 7)


def test_observer_accepts_any_percept_size():
    assert Observer(3)(torch.rand(2, 1, 64, 48)).shape == (2, 3)


def test_observer_learns_a_separable_task():
    x, y = _toy_task()
    observer = train_observer(x, y, n_classes=2, epochs=8, seed=0)
    x_test, y_test = _toy_task(seed=1)
    assert accuracy(observer, x_test, y_test) > 0.95


def test_training_is_reproducible_with_a_seed():
    x, y = _toy_task(n=64)
    a = train_observer(x, y, n_classes=2, epochs=1, seed=3)
    b = train_observer(x, y, n_classes=2, epochs=1, seed=3)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)


def test_chance_level_on_random_labels():
    x, _ = _toy_task(n=256)
    y = torch.randint(0, 4, (256,), generator=torch.Generator().manual_seed(9))
    observer = train_observer(x[:128], y[:128], n_classes=4, epochs=3, seed=0)
    assert accuracy(observer, x[128:], y[128:]) < 0.5
