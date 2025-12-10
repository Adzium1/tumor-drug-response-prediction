from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def leave_cell_line_out_split(
    labels_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[pd.Index, pd.Index]:
    rng = np.random.default_rng(seed)
    cell_lines = labels_df["cell_line"].unique()
    rng.shuffle(cell_lines)
    n_test = max(1, int(len(cell_lines) * test_frac))
    test_cell_lines = set(cell_lines[:n_test])
    test_idx = labels_df.index[labels_df["cell_line"].isin(test_cell_lines)]
    train_idx = labels_df.index.difference(test_idx)
    return train_idx, test_idx


def tissue_holdout_split(
    labels_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    tissue_name: str,
) -> tuple[pd.Index, pd.Index]:
    if metadata_df is None:
        raise ValueError("Metadata is required for tissue holdout split.")
    if "tissue" not in metadata_df.columns:
        raise ValueError("Metadata must contain a 'tissue' column for tissue holdout split.")
    test_cell_lines = metadata_df.loc[metadata_df["tissue"] == tissue_name, "cell_line"].unique()
    test_idx = labels_df.index[labels_df["cell_line"].isin(test_cell_lines)]
    train_idx = labels_df.index.difference(test_idx)
    return train_idx, test_idx


def kfold_cell_line_splits(
    labels_df: pd.DataFrame,
    metadata_df: Optional[pd.DataFrame],
    k: int,
    seed: int = 42,
) -> list[tuple[pd.Index, pd.Index]]:
    rng = np.random.default_rng(seed)
    cell_lines = labels_df["cell_line"].unique()
    rng.shuffle(cell_lines)
    folds = np.array_split(cell_lines, k)
    splits = []
    for i in range(k):
        val_cls = set(folds[i])
        val_idx = labels_df.index[labels_df["cell_line"].isin(val_cls)]
        train_idx = labels_df.index.difference(val_idx)
        splits.append((train_idx, val_idx))
    return splits


# if __name__ == "__main__":
#     # Simple sanity check could be added here.
#     pass
