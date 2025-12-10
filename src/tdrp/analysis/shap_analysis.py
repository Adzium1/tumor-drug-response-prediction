from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import shap
from typing import Optional

from tdrp.models.fusion import TDRPModel


class CombinedModelWrapper:
    def __init__(self, model: TDRPModel, device: torch.device, omics_dim: int, drug_dim: int):
        self.model = model
        self.device = device
        self.omics_dim = omics_dim
        self.drug_dim = drug_dim

    def __call__(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            omics = torch.from_numpy(X[:, : self.omics_dim]).float().to(self.device)
            drug_fp = torch.from_numpy(X[:, self.omics_dim :]).float().to(self.device)
            preds, _, _ = self.model(omics, drug_fp)
            return preds.cpu().numpy()


def _build_feature_matrix(
    omics_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    indices: np.ndarray,
) -> np.ndarray:
    omics_idx = omics_df.set_index("cell_line")
    drug_idx = drug_df.set_index("drug")
    rows = []
    for idx in indices:
        row = labels_df.iloc[idx]
        omics = omics_idx.loc[row["cell_line"]].values
        drug_fp = drug_idx.loc[row["drug"]].values
        rows.append(np.concatenate([omics, drug_fp]))
    return np.stack(rows)


def compute_shap_values(
    model: TDRPModel,
    omics_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    sample_size: int = 1000,
    background_size: int = 100,
    device: str = "cpu",
) -> dict:
    rng = np.random.default_rng(0)
    n_samples = min(sample_size, len(labels_df))
    sample_indices = rng.choice(len(labels_df), size=n_samples, replace=False)
    background_indices = rng.choice(len(labels_df), size=min(background_size, len(labels_df)), replace=False)

    X_sample = _build_feature_matrix(omics_df, drug_df, labels_df, sample_indices)
    X_background = _build_feature_matrix(omics_df, drug_df, labels_df, background_indices)

    device_t = torch.device(device)
    wrapper = CombinedModelWrapper(model, device_t, omics_df.shape[1] - 1, drug_df.shape[1] - 1)
    explainer = shap.KernelExplainer(wrapper, X_background)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    feature_names = list(omics_df.columns[1:]) + list(drug_df.columns[1:])
    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": feature_names,
        "sample_indices": sample_indices,
    }
