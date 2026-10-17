"""Experiment config loading shared by all scripts."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, quick: bool = False) -> dict:
    """Load a YAML experiment config; with ``quick``, merge the ``quick:`` section's overrides.

    A mapping under ``quick`` merges into the section of the same name, e.g.
    ``quick: {array: {n_electrodes: 64}}``; any other value replaces the top-level key, e.g.
    ``quick: {output: results/quick/fig.png}`` so smoke runs never overwrite real results.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    if quick:
        for key, override in cfg.get("quick", {}).items():
            if isinstance(override, dict):
                cfg[key].update(override)
            else:
                cfg[key] = override
    return cfg
