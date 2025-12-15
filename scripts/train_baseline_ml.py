"""
Lightweight baseline training using scikit-learn on processed data and split CSVs.

Steps:
1) Load processed tables: omics.parquet, drug_fingerprints.parquet, labels.parquet.
2) Load a split CSV produced by scripts/make_splits.py with columns
   [cell_line, drug, ln_ic50, split].
3) Fit PCA transformers on omics and drug fingerprints using the train split only.
4) Train an MLP regressor on the concatenated PCA features.
5) Report RMSE and Pearson correlation on train/val/test splits.

Usage:
    python scripts/train_baseline_ml.py \
        --processed-dir data/processed \
        --split-csv data/processed/splits/random_pair_split.csv \
        --outdir outputs/baseline_random

You can re-run with different split CSVs (cellline_holdout_split.csv, tissue_holdout_split.csv)
to test generalization.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    vx = x - x.mean()
    vy = y - y.mean()
    denom = np.sqrt((vx ** 2).sum()) * np.sqrt((vy ** 2).sum())
    if denom == 0:
        return np.nan
    return float((vx * vy).sum() / denom)


def load_processed(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    omics = pd.read_parquet(processed_dir / "omics.parquet")
    drugs = pd.read_parquet(processed_dir / "drug_fingerprints.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")[["cell_line", "drug", "ln_ic50"]]
    return {"omics": omics, "drugs": drugs, "labels": labels}


def build_features(
    omics_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    split_df: pd.DataFrame,
    pca_omics: PCA,
    pca_drug: PCA,
    scaler_omics: StandardScaler,
    scaler_drug: StandardScaler,
) -> (np.ndarray, np.ndarray):
    """Create concatenated PCA features for a subset defined by split_df."""
    cells = split_df["cell_line"].values
    drugs = split_df["drug"].values
    y = split_df["ln_ic50"].values.astype(np.float32)

    omics_mat = omics_df.set_index("cell_line").loc[cells].to_numpy(dtype=np.float32)
    drug_mat = drug_df.set_index("drug").loc[drugs].to_numpy(dtype=np.float32)

    omics_z = scaler_omics.transform(omics_mat)
    drug_z = scaler_drug.transform(drug_mat)

    omics_p = pca_omics.transform(omics_z)
    drug_p = pca_drug.transform(drug_z)

    X = np.concatenate([omics_p, drug_p], axis=1).astype(np.float32)
    return X, y


def train_one_split(processed_dir: Path, split_csv: Path, outdir: Path, n_omics_pca: int, n_drug_pca: int):
    outdir.mkdir(parents=True, exist_ok=True)
    tables = load_processed(processed_dir)
    split_df = pd.read_csv(split_csv)

    # Harmonize key types
    tables["omics"]["cell_line"] = tables["omics"]["cell_line"].astype(str)
    tables["drugs"]["drug"] = tables["drugs"]["drug"].astype(str)
    for col in ["cell_line", "drug"]:
        split_df[col] = split_df[col].astype(str)

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    omics_mat_train = tables["omics"].set_index("cell_line").loc[train_df["cell_line"]].to_numpy(dtype=np.float32)
    drug_mat_train = tables["drugs"].set_index("drug").loc[train_df["drug"]].to_numpy(dtype=np.float32)

    scaler_omics = StandardScaler().fit(omics_mat_train)
    scaler_drug = StandardScaler().fit(drug_mat_train)

    pca_omics = PCA(n_components=min(n_omics_pca, omics_mat_train.shape[1]), random_state=42).fit(
        scaler_omics.transform(omics_mat_train)
    )
    pca_drug = PCA(n_components=min(n_drug_pca, drug_mat_train.shape[1]), random_state=42).fit(
        scaler_drug.transform(drug_mat_train)
    )

    X_train, y_train = build_features(
        tables["omics"], tables["drugs"], train_df, pca_omics, pca_drug, scaler_omics, scaler_drug
    )
    X_val, y_val = build_features(
        tables["omics"], tables["drugs"], val_df, pca_omics, pca_drug, scaler_omics, scaler_drug
    )
    X_test, y_test = build_features(
        tables["omics"], tables["drugs"], test_df, pca_omics, pca_drug, scaler_omics, scaler_drug
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=50,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
    )
    mlp.fit(X_train, y_train)

    def eval_split(name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = mlp.predict(X)
        rmse = float(np.sqrt(np.mean((y - preds) ** 2)))
        r = pearsonr(y, preds)
        return {"rmse": rmse, "pearson": r}

    metrics = {
        "train": eval_split("train", X_train, y_train),
        "val": eval_split("val", X_val, y_val),
        "test": eval_split("test", X_test, y_test),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "omics_pca": int(pca_omics.n_components_),
        "drug_pca": int(pca_drug.n_components_),
    }

    pd.DataFrame([metrics]).to_json(outdir / "metrics.json", orient="records", lines=True)
    print(f"Split: {split_csv.name}")
    for split in ["train", "val", "test"]:
        m = metrics[split]
        print(f"{split}: RMSE={m['rmse']:.3f} Pearson={m['pearson']:.3f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train a simple PCA+MLP baseline on a split CSV.")
    ap.add_argument("--processed-dir", default="data/processed", help="Directory with parquet files.")
    ap.add_argument("--split-csv", required=True, help="Path to split CSV (random_pair_split.csv, etc.)")
    ap.add_argument("--outdir", required=True, help="Output directory for metrics.json")
    ap.add_argument("--omics-pca", type=int, default=128, help="PCA components for omics.")
    ap.add_argument("--drug-pca", type=int, default=128, help="PCA components for drug fingerprints.")
    args = ap.parse_args()

    train_one_split(
        processed_dir=Path(args.processed_dir),
        split_csv=Path(args.split_csv),
        outdir=Path(args.outdir),
        n_omics_pca=args.omics_pca,
        n_drug_pca=args.drug_pca,
    )


if __name__ == "__main__":
    main()
