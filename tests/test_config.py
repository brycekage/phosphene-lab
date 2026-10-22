"""Tests for experiment config loading."""

from phosphene_lab.config import load_config, with_overrides

CONFIG = """
seed: 3
output: results/fig.png
array: {layout: grid, n_electrodes: 256}
quick:
  output: results/quick/fig.png
  array: {n_electrodes: 64}
"""


def test_load_config_without_quick_ignores_overrides(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(CONFIG)
    cfg = load_config(path)
    assert cfg["array"] == {"layout": "grid", "n_electrodes": 256}


def test_quick_overrides_merge_into_sections(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(CONFIG)
    cfg = load_config(path, quick=True)
    assert cfg["array"] == {"layout": "grid", "n_electrodes": 64}
    assert cfg["seed"] == 3


def test_quick_scalar_overrides_replace_top_level_values(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(CONFIG)
    assert load_config(path)["output"] == "results/fig.png"
    assert load_config(path, quick=True)["output"] == "results/quick/fig.png"


def test_per_task_overrides_change_only_the_named_task():
    base = {"steps": 1500, "power_weight": 0.05}
    overrides = {"obstacles": {"power_weight": 0.005}}
    assert with_overrides(base, overrides, "obstacles") == {"steps": 1500, "power_weight": 0.005}
    assert with_overrides(base, overrides, "letters") == base
    assert with_overrides(base, None, "obstacles") == base
    assert base["power_weight"] == 0.05  # the shared settings are not mutated
