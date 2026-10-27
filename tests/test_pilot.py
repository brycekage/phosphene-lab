import json

import numpy as np
import pytest

from phosphene_lab.demo.pilot import Trial, choose_images, load_trials

ENCODERS = ("E0", "E2", "E3")


def test_choose_images_is_balanced_and_never_repeats_an_image():
    labels = np.repeat(np.arange(4), 20)
    rng = np.random.default_rng(0)
    picks = choose_images(labels, ENCODERS, classes=[0, 1, 2, 3], per_class=2, rng=rng)
    assert set(picks) == set(ENCODERS)
    every = [i for idx in picks.values() for i in idx]
    assert len(every) == len(set(every)) == 3 * 4 * 2
    for idx in picks.values():
        assert sorted(np.bincount(labels[idx], minlength=4)) == [2, 2, 2, 2]


def test_choose_images_skips_excluded_and_fails_loudly_when_short():
    labels = np.repeat(np.arange(2), 4)
    rng = np.random.default_rng(0)
    picks = choose_images(labels, ENCODERS, [0], 1, rng, exclude={0})
    assert 0 not in [i for idx in picks.values() for i in idx]
    with pytest.raises(ValueError, match="class 1"):
        choose_images(labels, ENCODERS, [1], 2, rng)


def test_load_trials_reads_the_manifest(tmp_path):
    trial = Trial("letters", "E0", "a", "letters/x.png", "letters/x_orig.png")
    manifest = {"classes": {"letters": ["a", "b"]}, "trials": [trial.__dict__]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    trials, classes = load_trials(tmp_path)
    assert trials == [trial]
    assert classes == {"letters": ["a", "b"]}
