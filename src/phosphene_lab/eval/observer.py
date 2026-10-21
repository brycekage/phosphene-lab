"""Virtual observer: a small CNN that tries to recognize what a percept shows.

It stands in for a patient: if a classifier trained on an encoder's percepts can tell letters
(or objects, or obstacle positions) apart, the percepts still carry that information. Each
encoder gets its own observer trained on its own percepts, so an encoder is judged by how
much task information survives, not by how well it matches one fixed reader.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Observer(nn.Module):
    def __init__(self, n_classes: int, width: int = 32) -> None:
        super().__init__()

        def block(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()
            )

        self.features = nn.Sequential(
            block(1, width // 2), nn.MaxPool2d(2),
            block(width // 2, width), nn.MaxPool2d(2),
            block(width, 2 * width), nn.AdaptiveAvgPool2d(4),
        )  # fmt: skip
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(32 * width, n_classes))

    def forward(self, percepts: Tensor) -> Tensor:
        """(B, 1, H, W) or (B, H, W) -> logits (B, n_classes)."""
        if percepts.dim() == 3:
            percepts = percepts.unsqueeze(1)
        return self.head(self.features(percepts))


def train_observer(
    percepts: Tensor,
    labels: Tensor,
    n_classes: int,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-3,
    seed: int = 0,
    width: int = 32,
) -> Observer:
    torch.manual_seed(seed)
    observer = Observer(n_classes, width)
    optimizer = torch.optim.Adam(observer.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    observer.train()
    for _ in range(epochs):
        order = torch.randperm(len(labels), generator=g)
        for start in range(0, len(labels), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) < 2:  # BatchNorm needs at least two samples
                continue
            loss = F.cross_entropy(observer(percepts[idx]), labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return observer.eval()


@torch.no_grad()
def accuracy(observer: Observer, percepts: Tensor, labels: Tensor, batch_size: int = 256) -> float:
    observer.eval()
    correct = 0
    for start in range(0, len(labels), batch_size):
        logits = observer(percepts[start : start + batch_size])
        correct += (logits.argmax(dim=1) == labels[start : start + batch_size]).sum().item()
    return correct / len(labels)
