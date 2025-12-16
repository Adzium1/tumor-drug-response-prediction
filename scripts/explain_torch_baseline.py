"""
Compute SHAP values for the Torch PCA+MLP baseline trained via train_torch_baseline.py.

Loads the saved checkpoint and PCA/scaler transforms, rebuilds features from a
split CSV, and runs Kernel SHAP on a sampled subset.

Example:
python scripts/explain_torch_baseline.py \
  --processed-dir data/processed \
  --split-csv data/processed/splits/random_pair_split.csv \
  --checkpoint outputs/torch_baseline_random/model.pt \
  --transforms outputs/torch_baseline_random/transforms.joblib \
  --metrics outputs/torch_baseline_random/metrics.json \
  --output outputs/torch_baseline_random/shap_values.npz \
  --sample-size 500 --background-size 100 --split test
"""

from __future__ import annotations

import os

# Avoid OpenMP runtime collision on some Windows/PyTorch + MKL stacks.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap
import torch
from joblib import load as joblib_load
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.models.torch_mlp import TorchMLP
from tdrp.utils.io import ensure_dir
from tdrp.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP for Torch PCA+MLP baseline.")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory with parquet feature tables.")
    parser.add_argument("--split-csv", required=True, help="Split CSV used for training (random/cell/tissue).")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt from train_torch_baseline.py.")
    parser.add_argument("--transforms", required=True, help="Path to transforms.joblib (scalers + PCA).")
    parser.add_argument("--metrics", help="Optional metrics.json to read hidden layers/dropout config.")
    parser.add_argument("--output", required=True, help="Output .npz path for SHAP values.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="Which split to explain.")
    parser.add_argument("--sample-size", type=int, default=500, help="Number of samples to explain.")
    parser.add_argument("--background-size", type=int, default=100, help="Background size for Kernel SHAP.")
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=None, help="Hidden layers if metrics.json not provided.")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout if metrics.json not provided.")
    parser.add_argument("--device", default="auto", help="Device string (cpu/cuda); auto picks cuda if available.")
    return parser.parse_args()


def load_processed(processed_dir: Path) -> Dict[str, pd.DataFrame]:
    omics = pd.read_parquet(processed_dir / "omics.parquet")
    drugs = pd.read_parquet(processed_dir / "drug_fingerprints.parquet")
    labels = pd.read_parquet(processed_dir / "labels.parquet")[["cell_line", "drug", "ln_ic50"]]
    return {"omics": omics, "drugs": drugs, "labels": labels}


def _build_features(
    df_split: pd.DataFrame,
    omics_idx: pd.DataFrame,
    drug_idx: pd.DataFrame,
    scaler_omics: StandardScaler,
    scaler_drug: StandardScaler,
    pca_omics: PCA,
    pca_drug: PCA,
) -> np.ndarray:
    cells = df_split["cell_line"].values
    drugs = df_split["drug"].values
    omics_mat = omics_idx.loc[cells].to_numpy(dtype=np.float32)
    drug_mat = drug_idx.loc[drugs].to_numpy(dtype=np.float32)
    omics_z = scaler_omics.transform(omics_mat)
    drug_z = scaler_drug.transform(drug_mat)
    omics_p = pca_omics.transform(omics_z)
    drug_p = pca_drug.transform(drug_z)
    return np.concatenate([omics_p, drug_p], axis=1).astype(np.float32)


class BaselineWrapper:
    """Callable wrapper for KernelExplainer."""

    def __init__(self, model: TorchMLP, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(X).float().to(self.device)
            preds = self.model(x_t)
            return preds.cpu().numpy()


def main() -> None:
    args = parse_args()
    setup_logging()
    ensure_dir(Path(args.output).parent)
    np.random.seed(42)
    torch.manual_seed(42)

    device_str = args.device
    if args.device == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logging.info("Using device: %s", device)

    tables = load_processed(Path(args.processed_dir))
    split_df = pd.read_csv(args.split_csv)
    required_cols = {"cell_line", "drug", "ln_ic50", "split"}
    missing_cols = required_cols - set(split_df.columns)
    if missing_cols:
        raise ValueError(f"Split CSV missing required columns: {missing_cols}")
    split_df["cell_line"] = split_df["cell_line"].astype(str)
    split_df["drug"] = split_df["drug"].astype(str)
    tables["omics"]["cell_line"] = tables["omics"]["cell_line"].astype(str)
    tables["drugs"]["drug"] = tables["drugs"]["drug"].astype(str)

    valid_mask = split_df["cell_line"].isin(tables["omics"]["cell_line"]) & split_df["drug"].isin(tables["drugs"]["drug"])
    if not valid_mask.all():
        logging.warning("Dropping %d rows from split CSV not found in processed tables.", (~valid_mask).sum())
        split_df = split_df[valid_mask]
    if args.split != "all":
        split_df = split_df[split_df["split"] == args.split]
    if split_df.empty:
        raise ValueError(f"No rows available for split '{args.split}'.")

    transforms = joblib_load(args.transforms)
    scaler_omics: StandardScaler = transforms["scaler_omics"]
    scaler_drug: StandardScaler = transforms["scaler_drug"]
    pca_omics: PCA = transforms["pca_omics"]
    pca_drug: PCA = transforms["pca_drug"]

    hidden_layers: Optional[List[int]] = None
    dropout: float = 0.3
    if args.metrics:
        with open(args.metrics, "r", encoding="utf-8") as f:
            metrics_obj = json.load(f)
        cfg = metrics_obj.get("config", {})
        hidden_layers = cfg.get("hidden_layers", None)
        dropout = cfg.get("dropout", dropout)
    if args.hidden_layers:
        hidden_layers = list(args.hidden_layers)
    if hidden_layers is None:
        hidden_layers = [512, 256]
    if args.dropout is not None:
        dropout = args.dropout

    input_dim = pca_omics.n_components_ + pca_drug.n_components_
    model = TorchMLP(input_dim=input_dim, hidden_layers=hidden_layers, dropout=dropout).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    omics_idx = tables["omics"].set_index("cell_line")
    drug_idx = tables["drugs"].set_index("drug")
    X_all = _build_features(split_df, omics_idx, drug_idx, scaler_omics, scaler_drug, pca_omics, pca_drug)

    rng = np.random.default_rng(0)
    n_samples = min(args.sample_size, len(X_all))
    sample_idx = rng.choice(len(X_all), size=n_samples, replace=False)
    bg_idx = rng.choice(len(X_all), size=min(args.background_size, len(X_all)), replace=False)

    X_sample = X_all[sample_idx]
    X_bg = X_all[bg_idx]

    wrapper = BaselineWrapper(model, device)
    explainer = shap.KernelExplainer(wrapper, X_bg)
    shap_values = explainer.shap_values(X_sample, nsamples=100)

    feature_names = [f"omics_pca_{i}" for i in range(pca_omics.n_components_)] + [
        f"drug_pca_{i}" for i in range(pca_drug.n_components_)
    ]
    np.savez(
        args.output,
        shap_values=shap_values,
        expected_value=explainer.expected_value,
        feature_names=feature_names,
        sample_indices=sample_idx,
        split=args.split,
    )
    logging.info("Saved SHAP values to %s", args.output)


if __name__ == "__main__":
    main()
