import json

import numpy as np
import pytest

from phosphene_lab.demo.pilot import PilotSession, Trial, choose_images, load_trials

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


def make_trials():
    trials = []
    for task, labels in (("letters", "abc"), ("obstacles", ["left", "none"])):
        trials += [Trial(task, "E0", labels[0], f"{task}_p.png", "o.png", practice=True)]
        trials += [
            Trial(task, enc, lab, f"{task}_{enc}_{lab}.png", "o.png")
            for enc in ENCODERS
            for lab in labels
        ]
    return trials


def test_session_runs_each_block_practice_first_then_shuffled_trials():
    session = PilotSession(make_trials(), blocks=("letters", "obstacles"), seed=1)
    order = session.order
    assert [t.task for t in order[:10]] == ["letters"] * 10
    assert order[0].practice and not any(t.practice for t in order[1:10])
    assert order[10].practice and order[10].task == "obstacles"
    assert len(order) == len(make_trials())
    other = PilotSession(make_trials(), blocks=("letters", "obstacles"), seed=2)
    assert [t.image for t in other.order] != [t.image for t in order]


def test_session_asks_for_an_intro_at_each_block_start():
    session = PilotSession(make_trials(), blocks=("letters", "obstacles"), seed=0)
    assert session.at_block_start
    session.start_block()
    assert not session.at_block_start
    for _ in range(10):
        session.answer("a")
    assert session.at_block_start and session.current.task == "obstacles"


def test_scoring_ignores_practice_and_counts_per_encoder():
    session = PilotSession(make_trials(), blocks=("letters", "obstacles"), seed=0)
    while not session.finished:
        trial = session.current
        correct = session.answer(trial.label if trial.encoder == "E2" else "zzz")
        assert correct == (trial.encoder == "E2")
        assert session.last == (trial, trial.label if trial.encoder == "E2" else "zzz", correct)
    summary = session.summary()
    assert summary["letters"] == {"E0": (0, 3), "E2": (3, 3), "E3": (0, 3)}
    assert summary["obstacles"]["E2"] == (2, 2)
    assert session.current is None


def test_progress_counts_scored_trials_within_the_block():
    session = PilotSession(make_trials(), blocks=("letters", "obstacles"), seed=0)
    session.start_block()
    assert session.progress() == (0, 9)  # practice does not count
    session.answer("a")  # the practice trial
    assert session.progress() == (0, 9)
    session.answer("a")
    assert session.progress() == (1, 9)


def test_load_trials_reads_the_manifest(tmp_path):
    trial = Trial("letters", "E0", "a", "letters/x.png", "letters/x_orig.png")
    manifest = {"classes": {"letters": ["a", "b"]}, "trials": [trial.__dict__]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    trials, classes = load_trials(tmp_path)
    assert trials == [trial]
    assert classes == {"letters": ["a", "b"]}
