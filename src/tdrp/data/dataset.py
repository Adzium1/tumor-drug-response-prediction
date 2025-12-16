from __future__ import annotations

from typing import Dict
import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    Minimal Dataset wrapper for precomputed tabular features and regression targets.

    Parameters
    ----------
    features : np.ndarray
        Array of shape (n_samples, n_features).
    targets : np.ndarray
        Array of shape (n_samples,) containing regression labels.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        if len(self.features) != len(self.targets):
            raise ValueError("features and targets must have the same length.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.features[idx]).float()
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return {"x": x, "y": y}
