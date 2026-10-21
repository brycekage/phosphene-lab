"""One benchmark cell: a virtual patient, an encoder, a task, and the metrics that come out.

The encoder works from the map the system *believes* (``encoder_map``: the true map by
default, a calibrated estimate in Phase 4), while percepts are always rendered with the
patient's *true* map, because that is what the patient actually sees. An observer is then
trained on this encoder's percepts of the training images and tested on held-out images.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from phosphene_lab.encoders.base import Encoder, image_to_field
from phosphene_lab.encoders.edges import EdgeEncoder
from phosphene_lab.encoders.learned import load_e3
from phosphene_lab.encoders.naive import NaiveEncoder
from phosphene_lab.encoders.scene import SceneEncoder
from phosphene_lab.eval.metrics import mean_ssim, stimulation_stats
from phosphene_lab.eval.observer import accuracy, train_observer
from phosphene_lab.eval.tasks import TASKS, load_task
from phosphene_lab.simulator.array import ElectrodeArray
from phosphene_lab.simulator.patient import PatientParams, VirtualPatient
from phosphene_lab.simulator.phosphene import PhospheneMap, PhospheneRenderer, RenderConfig

# How much patients differ from the nominal retinotopic prediction and from each other.
VARIABILITY = {
    "low": PatientParams(location_jitter_mm=0.5, threshold_log_sd=0.2, size_scale_log_sd=0.1),
    "default": PatientParams(),
    "high": PatientParams(location_jitter_mm=2.0, threshold_log_sd=0.6, size_scale_log_sd=0.3),
}


@dataclass(frozen=True)
class TaskData:
    train_images: Tensor
    train_labels: Tensor
    train_targets: Tensor
    test_images: Tensor
    test_labels: Tensor
    test_targets: Tensor
    n_classes: int


def load_task_data(name: str, root: str, limits: dict) -> TaskData:
    """Train and test images, labels and E2 targets for one task, capped by ``limits``."""
    task = TASKS[name]
    train = load_task(task, "train", root, limits["train"])
    test = load_task(task, "test", root, limits["test"])
    return TaskData(*train, *test, n_classes=task.n_classes)


def build_encoders(
    names: list[str], mode: str, task_name: str, render: RenderConfig, checkpoint_dir: str
) -> dict[str, Encoder]:
    """E0..E3 for one task. E2 needs no networks here: the benchmark feeds it cached targets."""
    factories = {
        "E0": lambda: NaiveEncoder(config=render),
        "E1": lambda: EdgeEncoder(config=render),
        "E2": lambda: SceneEncoder(mode, models=None, config=render),
        "E3": lambda: load_e3(str(Path(checkpoint_dir) / f"e3_{task_name}.pt"), render)[0],
    }
    return {name: factories[name]() for name in names}


def make_patient(n_electrodes: int, dropout: float, variability: str, seed: int) -> VirtualPatient:
    if variability not in VARIABILITY:
        raise ValueError(f"unknown variability {variability!r}; choose from {list(VARIABILITY)}")
    params = dataclasses.replace(VARIABILITY[variability], dropout_rate=dropout)
    return VirtualPatient(ElectrodeArray.grid(n_electrodes), params, seed=seed)


@torch.no_grad()
def encode(
    encoder: Encoder, images: Tensor, targets: Tensor, pmap: PhospheneMap, batch_size: int = 256
) -> Tensor:
    """Currents for a whole image set. E2 reuses its cached targets instead of re-running
    the depth and segmentation networks."""
    out = []
    for i in range(0, len(images), batch_size):
        if isinstance(encoder, SceneEncoder):
            out.append(encoder.encode_targets(targets[i : i + batch_size], pmap))
        else:
            out.append(encoder(images[i : i + batch_size], pmap))
    return torch.cat(out)


@torch.no_grad()
def render_all(renderer: PhospheneRenderer, stim: Tensor, batch_size: int = 256) -> Tensor:
    return torch.cat(
        [renderer.render(stim[i : i + batch_size]) for i in range(0, len(stim), batch_size)]
    )


def evaluate_cell(
    data: TaskData,
    encoder: Encoder,
    patient: VirtualPatient,
    render: RenderConfig,
    observer_epochs: int,
    seed: int = 0,
    encoder_map: PhospheneMap | None = None,
    observer_width: int = 32,
) -> dict[str, float]:
    true_map = patient.true_phosphene_map()
    belief = true_map if encoder_map is None else encoder_map
    renderer = PhospheneRenderer(true_map, render)

    train_stim = encode(encoder, data.train_images, data.train_targets, belief)
    test_stim = encode(encoder, data.test_images, data.test_targets, belief)
    train_percepts = render_all(renderer, train_stim)
    test_percepts = render_all(renderer, test_stim)

    observer = train_observer(
        train_percepts, data.train_labels, data.n_classes, epochs=observer_epochs, seed=seed,
        width=observer_width,
    )  # fmt: skip
    # The goal image sits where a perfectly calibrated system would put it.
    goal = image_to_field(data.test_targets, true_map, render)[:, 0]
    return {
        "accuracy": accuracy(observer, test_percepts, data.test_labels),
        "ssim": mean_ssim(test_percepts, goal),
        **stimulation_stats(test_stim),
    }
