from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


logger = logging.getLogger(__name__)


class PairDataset(Dataset):
    def __init__(
        self,
        omics_df: pd.DataFrame,
        drug_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ):
        """
        omics_df: DataFrame with 'cell_line' + features.
        drug_df: DataFrame with 'drug' + fingerprint bits.
        labels_df: DataFrame with columns ['cell_line', 'drug', 'ln_ic50'].
        """
        self.labels = labels_df.reset_index(drop=True).copy()
        self.labels["cell_line"] = self.labels["cell_line"].astype(str)
        self.labels["drug"] = self.labels["drug"].astype(str)

        omics_df = omics_df.copy()
        omics_df["cell_line"] = omics_df["cell_line"].astype(str)
        self.omics_df = omics_df.set_index("cell_line")

        drug_df = drug_df.copy()
        drug_df["drug"] = drug_df["drug"].astype(str)
        self.drug_df = drug_df.set_index("drug")

        missing_omics = set(self.labels["cell_line"]) - set(self.omics_df.index)
        missing_drugs = set(self.labels["drug"]) - set(self.drug_df.index)
        if missing_omics:
            raise ValueError(f"Missing omics features for cell lines: {missing_omics}")
        if missing_drugs:
            raise ValueError(f"Missing drug features for drugs: {missing_drugs}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        row = self.labels.iloc[idx]
        cell_line = row["cell_line"]
        drug = row["drug"]
        y = float(row["ln_ic50"])

        omics = self.omics_df.loc[cell_line].values.astype("float32")
        drug_fp = self.drug_df.loc[drug].values.astype("float32")

        return {
            "omics": torch.from_numpy(omics),
            "drug_fp": torch.from_numpy(drug_fp),
            "y": torch.tensor(y, dtype=torch.float32),
        }


def make_dataloaders(
    omics_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    train_idx,
    val_idx,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    train_labels = labels_df.iloc[train_idx]
    val_labels = labels_df.iloc[val_idx]
    train_ds = PairDataset(omics_df, drug_df, train_labels)
    val_ds = PairDataset(omics_df, drug_df, val_labels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def _append_enriched_drug_features(
    drug_df: pd.DataFrame,
    enriched_path: Optional[str],
    numeric_columns: Optional[Iterable[str]],
) -> pd.DataFrame:
    if not enriched_path or not numeric_columns:
        return drug_df
    path = Path(enriched_path)
    if not path.exists():
        logger.warning("Enriched drug metadata not found at %s; skipping.", path)
        return drug_df

    enriched_df = pd.read_parquet(path)
    if "drug" not in enriched_df.columns:
        logger.warning("Enriched metadata missing 'drug' column; skipping.")
        return drug_df

    numeric_cols = [col for col in numeric_columns if col in enriched_df.columns]
    if not numeric_cols:
        logger.warning("No enriched numeric columns found in %s; skipping.", path)
        return drug_df

    extra = enriched_df[["drug", *numeric_cols]].copy()
    extra["drug"] = extra["drug"].astype(str)
    for col in numeric_cols:
        extra[col] = pd.to_numeric(extra[col], errors="coerce")

    means = extra[numeric_cols].mean()
    stds = extra[numeric_cols].std().replace(0, 1)
    extra[numeric_cols] = (extra[numeric_cols].fillna(means) - means) / stds

    merged = drug_df.copy()
    merged["drug"] = merged["drug"].astype(str)
    merged = merged.merge(extra, on="drug", how="left")
    merged[numeric_cols] = merged[numeric_cols].fillna(0.0)
    return merged


def load_processed_tables(
    processed_dir: str,
    use_enriched_drug_features: bool = False,
    enriched_drug_metadata_path: Optional[str] = None,
    enriched_numeric_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience loader for the processed parquet tables.

    Args:
        processed_dir: Directory containing omics.parquet, drug_fingerprints.parquet, labels.parquet.

    Returns:
        (omics_df, drug_df, labels_df) dataframes with a leading key column:
        - omics_df: 'cell_line' + expression features
        - drug_df: 'drug' + fingerprint bits
        - labels_df: ['cell_line', 'drug', 'ln_ic50']
    """
    base = Path(processed_dir)
    omics = pd.read_parquet(base / "omics.parquet")
    drugs = pd.read_parquet(base / "drug_fingerprints.parquet")
    labels = pd.read_parquet(base / "labels.parquet")
    labels = labels[["cell_line", "drug", "ln_ic50"]]
    if use_enriched_drug_features:
        drugs = _append_enriched_drug_features(drugs, enriched_drug_metadata_path, enriched_numeric_columns)
    return omics, drugs, labels


def make_loaders_from_split_csv(
    processed_dir: str,
    split_csv: str,
    batch_size: int,
    num_workers: int,
    splits: Iterable[str] = ("train", "val", "test"),
    use_enriched_drug_features: bool = False,
    enriched_drug_metadata_path: Optional[str] = None,
    enriched_numeric_columns: Optional[Iterable[str]] = None,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders directly from processed parquet files and a split CSV.

    The split CSV is produced by scripts/make_splits.py and must contain
    ['cell_line', 'drug', 'ln_ic50', 'split'] columns.

    Args:
        processed_dir: Directory with omics/drug_fingerprints/labels parquet files.
        split_csv: Path to a CSV with split assignments.
        batch_size: Batch size for all loaders.
        num_workers: Number of PyTorch workers.
        splits: Iterable of split names to return (default train/val/test).

    Returns:
        Dict mapping split name -> DataLoader yielding dicts with
        'omics', 'drug_fp', and 'y' (ln_ic50) tensors.
    """
    omics_df, drug_df, _ = load_processed_tables(
        processed_dir,
        use_enriched_drug_features=use_enriched_drug_features,
        enriched_drug_metadata_path=enriched_drug_metadata_path,
        enriched_numeric_columns=enriched_numeric_columns,
    )
    split_df = pd.read_csv(split_csv)

    loaders: Dict[str, DataLoader] = {}
    for split_name in splits:
        split_labels = split_df[split_df["split"] == split_name][["cell_line", "drug", "ln_ic50"]]
        if split_labels.empty:
            continue
        ds = PairDataset(omics_df, drug_df, split_labels)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )
    return loaders


def make_loaders_from_split_df(
    omics_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    split_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    splits: Iterable[str] = ("train", "val", "test"),
) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split_name in splits:
        split_labels = split_df[split_df["split"] == split_name][["cell_line", "drug", "ln_ic50"]]
        if split_labels.empty:
            continue
        ds = PairDataset(omics_df, drug_df, split_labels)
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )
    return loaders
