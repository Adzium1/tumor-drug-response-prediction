from __future__ import annotations

from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


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
        self.labels = labels_df.reset_index(drop=True)
        self.omics_df = omics_df.set_index("cell_line")
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
