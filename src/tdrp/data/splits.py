from __future__ import annotations

from typing import Optional, Sequence
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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


def filter_entities(
    df: pd.DataFrame,
    min_drug_frac: float = 0.7,
    min_cell_frac: float = 0.6,
    tissue_min_cells: Optional[int] = None,
    tissue_col: str = "tissue",
) -> pd.DataFrame:
    """
    Iteratively drop drugs and cell lines with too few measurements.

    A drug is kept if it has measurements for at least `min_drug_frac` of the
    remaining cell lines. A cell line is kept if it has measurements for at
    least `min_cell_frac` of the remaining drugs. The procedure repeats until
    convergence. Optionally drop tissues with fewer than `tissue_min_cells`
    distinct cell lines.
    """
    required_cols = {"cell_line", "drug"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")
    filtered = df.copy()
    while True:
        n_cells = filtered["cell_line"].nunique()
        n_drugs = filtered["drug"].nunique()
        drug_frac = filtered.groupby("drug")["cell_line"].nunique() / max(n_cells, 1)
        cell_frac = filtered.groupby("cell_line")["drug"].nunique() / max(n_drugs, 1)
        keep_drugs = set(drug_frac[drug_frac >= min_drug_frac].index)
        keep_cells = set(cell_frac[cell_frac >= min_cell_frac].index)
        new_filtered = filtered[filtered["drug"].isin(keep_drugs) & filtered["cell_line"].isin(keep_cells)]
        if len(new_filtered) == len(filtered):
            break
        filtered = new_filtered
    if tissue_min_cells is not None and tissue_col in filtered.columns:
        tissue_counts = (
            filtered[["cell_line", tissue_col]].drop_duplicates().groupby(tissue_col)["cell_line"].nunique()
        )
        keep_tissues = set(tissue_counts[tissue_counts >= tissue_min_cells].index)
        dropped = set(tissue_counts.index) - keep_tissues
        if dropped:
            logger.info("Dropping tissues with < %d cell lines: %s", tissue_min_cells, sorted(dropped))
        filtered = filtered[filtered[tissue_col].isin(keep_tissues)]
    logger.info(
        "After filtering: %d pairs, %d cell lines, %d drugs%s",
        len(filtered),
        filtered["cell_line"].nunique(),
        filtered["drug"].nunique(),
        f", tissues={filtered[tissue_col].nunique()}" if tissue_col in filtered.columns else "",
    )
    return filtered.reset_index(drop=True)


def make_random_pair_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    split_col: str = "split",
    stratify_by_tissue: bool = True,
    tissue_col: str = "tissue",
) -> pd.DataFrame:
    """
    Random split of observed pairs with optional tissue stratification.

    If `stratify_by_tissue` and the tissue column is present, each tissue slice is
    split independently to maintain tissue proportions (important when PCA shows
    strong tissue structure). Otherwise falls back to stratifying within each drug.
    """
    rng = np.random.default_rng(seed)
    result = df.copy()
    result[split_col] = ""
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    def _assign(idx: np.ndarray):
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        n_test = n - n_train - n_val
        if n_test == 0 and n > 0:
            n_test = 1
            n_train = max(1, n_train - 1)
        result.loc[idx[:n_train], split_col] = "train"
        result.loc[idx[n_train : n_train + n_val], split_col] = "val"
        result.loc[idx[n_train + n_val :], split_col] = "test"

    if stratify_by_tissue and tissue_col in result.columns:
        for tissue, sub in result.groupby(tissue_col):
            _assign(sub.index.to_numpy())
    else:
        if "drug" not in result.columns:
            raise ValueError("Input DataFrame must contain a 'drug' column when not stratifying by tissue.")
        for _, sub in result.groupby("drug"):
            _assign(sub.index.to_numpy())

    _log_split_summary(result, split_col, context="random pair-wise")
    return result


def make_cellline_holdout_split(
    df: pd.DataFrame,
    tissue_col: str = "tissue",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    split_col: str = "split",
    min_tissue_cells: int = 25,
) -> pd.DataFrame:
    """
    Stratified split where cell lines (not pairs) are held out, grouped by tissue.

    For each tissue with at least `min_tissue_cells` cell lines, assign cell lines
    into train/val/test (70/15/15 by default). All pairs for those cell lines
    follow their split assignment.
    """
    required_cols = {"cell_line", tissue_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing columns: {missing}")
    rng = np.random.default_rng(seed)
    result = df[df[tissue_col].notna()].copy()
    result[split_col] = ""
    eligible_tissues = (
        result[["cell_line", tissue_col]].drop_duplicates().groupby(tissue_col)["cell_line"].nunique()
    )
    eligible_tissues = eligible_tissues[eligible_tissues >= min_tissue_cells].index.tolist()
    result = result[result[tissue_col].isin(eligible_tissues)]
    test_frac = 1.0 - train_frac - val_frac
    for tissue in eligible_tissues:
        cells = result.loc[result[tissue_col] == tissue, "cell_line"].unique()
        rng.shuffle(cells)
        n = len(cells)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        train_cells = set(cells[:n_train])
        val_cells = set(cells[n_train : n_train + n_val])
        test_cells = set(cells[n_train + n_val :])
        result.loc[result["cell_line"].isin(train_cells), split_col] = "train"
        result.loc[result["cell_line"].isin(val_cells), split_col] = "val"
        result.loc[result["cell_line"].isin(test_cells), split_col] = "test"
    _log_split_summary(result, split_col, context="cell-line holdout")
    return result


def make_tissue_holdout_split(
    df: pd.DataFrame,
    tissue_col: str = "tissue",
    test_tissues: Sequence[str] = (),
    train_frac: float = 0.8,
    seed: int = 42,
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Hold out selected tissues entirely; split remaining tissues' cell lines into train/val.

    - All pairs from tissues in `test_tissues` are assigned to the test split.
    - Remaining tissues' cell lines are split train/val (e.g., 80/20).
    """
    if tissue_col not in df.columns:
        raise ValueError(f"Input DataFrame must contain a '{tissue_col}' column.")
    if not test_tissues:
        raise ValueError("test_tissues must be a non-empty sequence.")
    rng = np.random.default_rng(seed)
    result = df.copy()
    result[split_col] = ""
    test_tissues_set = set(test_tissues)
    result.loc[result[tissue_col].isin(test_tissues_set), split_col] = "test"
    remaining = result[result[split_col] != "test"]
    remaining_tissues = remaining[tissue_col].dropna().unique()
    for tissue in remaining_tissues:
        cells = remaining.loc[remaining[tissue_col] == tissue, "cell_line"].unique()
        rng.shuffle(cells)
        n = len(cells)
        n_train = max(1, int(n * train_frac))
        n_val = n - n_train
        train_cells = set(cells[:n_train])
        val_cells = set(cells[n_train:])
        result.loc[result["cell_line"].isin(train_cells) & (result[tissue_col] == tissue), split_col] = "train"
        result.loc[result["cell_line"].isin(val_cells) & (result[tissue_col] == tissue), split_col] = "val"
    _log_split_summary(result, split_col, context="tissue holdout")
    return result


def _log_split_summary(df: pd.DataFrame, split_col: str, context: str) -> None:
    counts = df[split_col].value_counts()
    cell_counts = df.groupby(split_col)["cell_line"].nunique()
    drug_counts = df.groupby(split_col)["drug"].nunique()
    tissue_counts = df.groupby(split_col)["tissue"].nunique() if "tissue" in df.columns else {}
    logger.info(
        "[%s] pairs per split: %s | cell lines: %s | drugs: %s%s",
        context,
        counts.to_dict(),
        cell_counts.to_dict(),
        drug_counts.to_dict(),
        f" | tissues: {tissue_counts.to_dict()}" if isinstance(tissue_counts, pd.Series) else "",
    )
