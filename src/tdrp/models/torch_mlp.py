from __future__ import annotations

from typing import Sequence
import torch
from torch import nn


class TorchMLP(nn.Module):
    """
    Simple feedforward regressor for concatenated PCA features.

    Hidden layers mirror the sklearn baseline (ReLU + Dropout).
    """

    def __init__(self, input_dim: int, hidden_layers: Sequence[int], dropout: float = 0.3):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
