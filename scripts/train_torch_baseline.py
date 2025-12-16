"""
Torch reimplementation of the sklearn PCA+MLP baseline for ln(IC50) prediction.

Mirrors scripts/train_baseline_ml.py: PCA on omics/drug fingerprints (fit on
train split), concatenate features, train a small MLP, and report RMSE/Pearson
on train/val/test splits defined in a split CSV.

Example:
python scripts/train_torch_baseline.py \
  --processed-dir data/processed \
  --split-csv data/processed/splits/random_pair_split.csv \
  --omics-pca 256 --drug-pca 128 \
  --hidden-layers 512 256 \
  --epochs 100 \
  --outdir outputs/torch_baseline_random
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from joblib import dump
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.data.dataset import TabularDataset
from tdrp.models.torch_mlp import TorchMLP
from tdrp.training.trainer import Trainer
from tdrp.training.metrics import rmse, pearsonr
from tdrp.utils.io import ensure_dir, save_json
from tdrp.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch PCA+MLP baseline for ln(IC50).")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory with parquet files.")
    parser.add_argument("--split-csv", required=True, help="Path to split CSV (random/cellline/tissue holdout).")
    parser.add_argument("--omics-pca", type=int, default=256, help="PCA components for omics.")
    parser.add_argument("--drug-pca", type=int, default=128, help="PCA components for drug fingerprints.")
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[512, 256],
        help="Hidden layer sizes, space-separated (e.g., 512 256).",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val RMSE.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", default="auto", help="Device string (cpu/cuda); 'auto' picks cuda if available.")
    parser.add_argument("--outdir", required=True, help="Output directory for model and metrics.")
    return parser.parse_args()


def load_processed_tables(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    omics = pd.read_parquet(processed_dir / "omics.parquet")
    drugs = pd.read_parquet(processed_dir / "drug_fingerprints.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")[["cell_line", "drug", "ln_ic50"]]
    return {"omics": omics, "drugs": drugs, "labels": labels}


def _fit_transforms(
    omics_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    split_df: pd.DataFrame,
    n_omics_pca: int,
    n_drug_pca: int,
) -> Tuple[StandardScaler, StandardScaler, PCA, PCA]:
    train_df = split_df[split_df["split"] == "train"]
    omics_idx = omics_df.set_index("cell_line")
    drug_idx = drug_df.set_index("drug")

    omics_train = omics_idx.loc[train_df["cell_line"]].to_numpy(dtype=np.float32)
    drug_train = drug_idx.loc[train_df["drug"]].to_numpy(dtype=np.float32)

    scaler_omics = StandardScaler().fit(omics_train)
    scaler_drug = StandardScaler().fit(drug_train)

    pca_omics = PCA(n_components=min(n_omics_pca, omics_train.shape[1]), random_state=42).fit(
        scaler_omics.transform(omics_train)
    )
    pca_drug = PCA(n_components=min(n_drug_pca, drug_train.shape[1]), random_state=42).fit(
        scaler_drug.transform(drug_train)
    )
    return scaler_omics, scaler_drug, pca_omics, pca_drug


def _build_features(
    df_split: pd.DataFrame,
    omics_idx: pd.DataFrame,
    drug_idx: pd.DataFrame,
    scaler_omics: StandardScaler,
    scaler_drug: StandardScaler,
    pca_omics: PCA,
    pca_drug: PCA,
) -> Tuple[np.ndarray, np.ndarray]:
    cells = df_split["cell_line"].values
    drugs = df_split["drug"].values
    y = df_split["ln_ic50"].values.astype(np.float32)

    omics_mat = omics_idx.loc[cells].to_numpy(dtype=np.float32)
    drug_mat = drug_idx.loc[drugs].to_numpy(dtype=np.float32)

    omics_z = scaler_omics.transform(omics_mat)
    drug_z = scaler_drug.transform(drug_mat)

    omics_p = pca_omics.transform(omics_z)
    drug_p = pca_drug.transform(drug_z)

    X = np.concatenate([omics_p, drug_p], axis=1).astype(np.float32)
    return X, y


def _make_loaders(
    split_df: pd.DataFrame,
    omics_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    scaler_omics: StandardScaler,
    scaler_drug: StandardScaler,
    pca_omics: PCA,
    pca_drug: PCA,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    omics_idx = omics_df.set_index("cell_line")
    drug_idx = drug_df.set_index("drug")

    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    for name, df_part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if df_part.empty:
            logging.warning("Split '%s' is empty; metrics will be NaN.", name)

    X_train, y_train = _build_features(train_df, omics_idx, drug_idx, scaler_omics, scaler_drug, pca_omics, pca_drug)
    X_val, y_val = _build_features(val_df, omics_idx, drug_idx, scaler_omics, scaler_drug, pca_omics, pca_drug)
    X_test, y_test = _build_features(test_df, omics_idx, drug_idx, scaler_omics, scaler_drug, pca_omics, pca_drug)

    train_loader = DataLoader(
        TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, val_loader, test_loader


def evaluate_split(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            preds.append(pred.cpu())
            targets.append(y.cpu())
    if not preds:
        return {"rmse": float("nan"), "pearson": float("nan")}
    y_pred = torch.cat(preds).numpy()
    y_true = torch.cat(targets).numpy()
    return {"rmse": rmse(y_true, y_pred), "pearson": pearsonr(y_true, y_pred)}


def main() -> None:
    args = parse_args()
    setup_logging()
    ensure_dir(args.outdir)
    np.random.seed(42)
    torch.manual_seed(42)

    device_str = args.device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logging.info("Using device: %s", device)

    tables = load_processed_tables(Path(args.processed_dir))
    split_df = pd.read_csv(args.split_csv)
    required_cols = {"cell_line", "drug", "ln_ic50", "split"}
    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        raise ValueError(f"Split CSV missing required columns: {missing_cols}")
    for col in ["cell_line", "drug"]:
        split_df[col] = split_df[col].astype(str)
    tables["omics"]["cell_line"] = tables["omics"]["cell_line"].astype(str)
    tables["drugs"]["drug"] = tables["drugs"]["drug"].astype(str)

    # Drop rows not present in processed tables to avoid KeyErrors.
    valid_mask = split_df["cell_line"].isin(tables["omics"]["cell_line"]) & split_df["drug"].isin(tables["drugs"]["drug"])
    dropped = len(split_df) - valid_mask.sum()
    if dropped:
        logging.warning("Dropping %d rows from split CSV not found in processed tables.", dropped)
    split_df = split_df[valid_mask]
    if split_df.empty:
        raise ValueError("No rows remain after aligning split CSV with processed tables.")

    scaler_omics, scaler_drug, pca_omics, pca_drug = _fit_transforms(
        tables["omics"], tables["drugs"], split_df, n_omics_pca=args.omics_pca, n_drug_pca=args.drug_pca
    )

    train_loader, val_loader, test_loader = _make_loaders(
        split_df,
        tables["omics"],
        tables["drugs"],
        scaler_omics,
        scaler_drug,
        pca_omics,
        pca_drug,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    input_dim = args.omics_pca + args.drug_pca
    model = TorchMLP(input_dim=input_dim, hidden_layers=args.hidden_layers, dropout=args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, device=device, scheduler=scheduler)
    history = trainer.fit(train_loader, val_loader, epochs=args.epochs, patience=args.patience)

    metrics = {
        "train": evaluate_split(model, train_loader, device),
        "val": evaluate_split(model, val_loader, device),
        "test": evaluate_split(model, test_loader, device),
        "config": {
            "omics_pca": args.omics_pca,
            "drug_pca": args.drug_pca,
            "hidden_layers": args.hidden_layers,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "split_csv": args.split_csv,
        },
    }
    for split_name, m in metrics.items():
        if split_name == "config":
            continue
        logging.info("%s - RMSE=%.3f Pearson=%.3f", split_name, m["rmse"], m["pearson"])

    torch.save(model.state_dict(), Path(args.outdir) / "model.pt")
    dump(
        {
            "scaler_omics": scaler_omics,
            "scaler_drug": scaler_drug,
            "pca_omics": pca_omics,
            "pca_drug": pca_drug,
        },
        Path(args.outdir) / "transforms.joblib",
    )
    save_json(metrics, Path(args.outdir) / "metrics.json")
    save_json([h.__dict__ for h in history], Path(args.outdir) / "history.json")
    logging.info("Saved model and metrics to %s", args.outdir)


if __name__ == "__main__":
    main()
