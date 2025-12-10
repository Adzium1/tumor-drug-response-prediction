from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    x = y_true - y_true.mean()
    y = y_pred - y_pred.mean()
    denom = np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)
