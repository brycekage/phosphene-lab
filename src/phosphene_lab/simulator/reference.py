"""Thin wrappers around the reference simulators (pulse2percept, dynaphos).

Our own simulator (Phase 1) is cross-checked against these, so each wrapper
returns a plain 2-D image in [0, 1] for side-by-side comparison.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch


@contextmanager
def _preserve_global_torch_state() -> Iterator[None]:
    """Undo dynaphos's process-wide side effects.

    dynaphos.utils.set_deterministic reseeds the global torch/NumPy RNGs and turns on
    torch.use_deterministic_algorithms(True), which makes some CUDA ops raise later in
    unrelated code (e.g. encoder training). Restore everything once the simulator is built.
    """
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_rng = np.random.get_state()  # noqa: NPY002 - dynaphos reseeds the legacy global RNG
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        yield
    finally:
        torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng)  # noqa: NPY002
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = cudnn_flags


def render_pulse2percept(cfg: dict[str, Any]) -> np.ndarray:
    """Render one percept with pulse2percept's cortical scoreboard model."""
    from pulse2percept.implants import cortex as p2p_implants
    from pulse2percept.models.cortex import ScoreboardModel

    implant = getattr(p2p_implants, cfg["implant"])()
    field = cfg["field_deg"]
    model = ScoreboardModel(
        implant,
        rho=cfg["rho_um"],
        xrange=(-field, field),
        yrange=(-field, field),
        step=cfg["step_deg"],
        verbose=False,
    ).build()
    stim = {name: cfg["amplitude_uA"] for name in implant.electrodes}
    image = model.predict_percept(stim).data[..., 0]
    return _normalize(image)


def load_dynaphos_simulator(cfg: dict[str, Any], seed: int):
    """Build a dynaphos GaussianSimulator with randomly sampled phosphene locations."""
    from dynaphos import cortex_models

    params = _load_dynaphos_params(cfg, seed)
    rng = np.random.default_rng(seed)
    coords = cortex_models.get_visual_field_coordinates_probabilistically(
        params, cfg["n_phosphenes"], rng
    )
    return _build_dynaphos(params, coords, rng)


def load_dynaphos_simulator_at(
    x_deg: np.ndarray, y_deg: np.ndarray, cfg: dict[str, Any], seed: int
):
    """Build a dynaphos GaussianSimulator with phosphenes at the given locations (deg).

    Note: dynaphos images put the bottom of the visual field in row 0.
    """
    from dynaphos.utils import Map

    params = _load_dynaphos_params(cfg, seed)
    if "field_deg" in cfg:
        params["run"]["view_angle"] = cfg["field_deg"]
    if cfg.get("uniform_thresholds", False):
        params["thresholding"]["activation_threshold_sd"] = 0.0
    coords = Map(x=np.asarray(x_deg, dtype=float), y=np.asarray(y_deg, dtype=float))
    return _build_dynaphos(params, coords, np.random.default_rng(seed))


def _load_dynaphos_params(cfg: dict[str, Any], seed: int) -> dict:
    from dynaphos import utils

    params = utils.load_params(cfg["params"])
    params["run"]["gpu"] = None  # CPU; dynaphos picks CUDA whenever gpu is set
    params["run"]["seed"] = seed
    params["run"]["resolution"] = [cfg["resolution"], cfg["resolution"]]
    return params


def _build_dynaphos(params: dict, coords, rng: np.random.Generator):
    from dynaphos.simulator import GaussianSimulator

    with _preserve_global_torch_state():
        return GaussianSimulator(params, coords, rng)


def render_dynaphos(cfg: dict[str, Any], seed: int) -> torch.Tensor:
    """Render one percept with dynaphos; returns a differentiable tensor."""
    sim = load_dynaphos_simulator(cfg, seed)
    amplitude = torch.full((sim.num_phosphenes,), cfg["amplitude_A"])
    frame = None
    for _ in range(cfg["n_frames"]):
        frame = sim(amplitude)
    return frame


def _normalize(image: np.ndarray) -> np.ndarray:
    peak = float(image.max())
    return image / peak if peak > 0 else image
