from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from tdrp.config import ExperimentConfig
from tdrp.data.splits import leave_cell_line_out_split, tissue_holdout_split, kfold_cell_line_splits
from tdrp.training.dataset import make_dataloaders, make_loaders_from_split_df, load_processed_tables
from tdrp.training.metrics import rmse, pearsonr
from tdrp.models.fusion import TDRPModel
from tdrp.models.vae import vae_loss
from tdrp.utils.seed import set_seed
from tdrp.utils.io import load_parquet, ensure_dir, save_json


logger = logging.getLogger(__name__)


def _load_processed_tables(data_cfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    base = Path(data_cfg.processed_dir)
    omics_df, drug_df, labels_df = load_processed_tables(
        str(base),
        use_enriched_drug_features=data_cfg.use_enriched_drug_features,
        enriched_drug_metadata_path=data_cfg.enriched_drug_metadata_path,
        enriched_numeric_columns=data_cfg.enriched_numeric_columns,
    )
    metadata_df = None
    if data_cfg.metadata_file:
        meta_path = base / data_cfg.metadata_file
        if meta_path.exists():
            metadata_df = load_parquet(meta_path)
    return omics_df, drug_df, labels_df, metadata_df


def _build_split(cfg: ExperimentConfig, labels_df: pd.DataFrame, metadata_df: Optional[pd.DataFrame]):
    strategy = cfg.training.split_strategy
    if strategy == "leave_cell_line_out":
        return leave_cell_line_out_split(labels_df, metadata_df, seed=cfg.training.seed)
    if strategy == "tissue_holdout":
        tissue = cfg.training.tissue_holdout
        if tissue is None:
            raise ValueError("tissue_holdout must be set for tissue_holdout split strategy.")
        return tissue_holdout_split(labels_df, metadata_df, tissue)
    if strategy == "kfold":
        return kfold_cell_line_splits(labels_df, metadata_df, k=cfg.training.k_folds, seed=cfg.training.seed)
    raise ValueError(f"Unknown split strategy: {strategy}")


def _train_one_epoch(model: TDRPModel, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        omics = batch["omics"].to(device)
        drug_fp = batch["drug_fp"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        preds, mu, logvar = model(omics, drug_fp)
        mse_loss = nn.functional.mse_loss(preds, y)
        if model.use_vae and mu is not None and logvar is not None and model.vae is not None:
            recon, _, _ = model.vae(omics)
            loss = mse_loss + model.lambda_vae * vae_loss(recon, omics, mu, logvar)
        else:
            loss = mse_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def _evaluate(model: TDRPModel, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    preds_list = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            omics = batch["omics"].to(device)
            drug_fp = batch["drug_fp"].to(device)
            y = batch["y"].to(device)
            preds, _, _ = model(omics, drug_fp)
            preds_list.append(preds.cpu().numpy())
            targets.append(y.cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds_list)
    return rmse(y_true, y_pred), pearsonr(y_true, y_pred)


def _load_split_csv(path: str) -> pd.DataFrame:
    split_df = pd.read_csv(path)
    required = {"cell_line", "drug", "ln_ic50", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Split CSV missing columns: {sorted(missing)}")
    return split_df


def train_model(cfg: ExperimentConfig, output_dir: str):
    set_seed(cfg.training.seed)
    ensure_dir(output_dir)
    logger.info("Loading processed data")
    omics_df, drug_df, labels_df, metadata_df = _load_processed_tables(cfg.data)
    device = torch.device(cfg.training.device if torch.cuda.is_available() or cfg.training.device == "cpu" else "cpu")

    # Adjust model input dimensions based on processed data
    cfg.model.omics_dim = omics_df.shape[1] - 1
    cfg.model.drug_dim = drug_df.shape[1] - 1

    if cfg.data.split_csv:
        logger.info("Using split CSV at %s", cfg.data.split_csv)
        split_df = _load_split_csv(cfg.data.split_csv)
        loaders = make_loaders_from_split_df(
            omics_df,
            drug_df,
            split_df,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )
        train_loader = loaders.get("train")
        val_loader = loaders.get("val")
        test_loader = loaders.get("test")
        if train_loader is None or val_loader is None:
            raise ValueError("Split CSV must contain both train and val splits.")

        history = []
        model = TDRPModel(cfg.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        best_rmse = float("inf")
        best_state = None
        for epoch in range(cfg.training.max_epochs):
            train_loss = _train_one_epoch(model, train_loader, optimizer, device)
            val_rmse, val_pearson = _evaluate(model, val_loader, device)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "val_pearson": val_pearson,
                }
            )
            logger.info(
                "Epoch %d - train_loss=%.4f val_rmse=%.4f val_pearson=%.4f",
                epoch,
                train_loss,
                val_rmse,
                val_pearson,
            )
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = model.state_dict()

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, Path(output_dir) / "model.pt")

        metrics = {}
        train_rmse, train_pearson = _evaluate(model, train_loader, device)
        metrics["train"] = {"rmse": train_rmse, "pearson": train_pearson}
        val_rmse, val_pearson = _evaluate(model, val_loader, device)
        metrics["val"] = {"rmse": val_rmse, "pearson": val_pearson}
        if test_loader is not None:
            test_rmse, test_pearson = _evaluate(model, test_loader, device)
            metrics["test"] = {"rmse": test_rmse, "pearson": test_pearson}
        save_json(metrics, Path(output_dir) / "metrics.json")
        save_json(history, Path(output_dir) / "training_history.json")
        logger.info("Training complete. Metrics saved to %s", output_dir)
        return

    splits = _build_split(cfg, labels_df, metadata_df)
    split_list = splits if isinstance(splits, list) else [splits]

    history = []
    for fold_idx, split in enumerate(split_list):
        if isinstance(splits, list):
            train_idx, val_idx = split
        else:
            train_idx, val_idx = split
        logger.info("Fold %d: %d train / %d val", fold_idx, len(train_idx), len(val_idx))
        train_loader, val_loader = make_dataloaders(
            omics_df,
            drug_df,
            labels_df,
            train_idx,
            val_idx,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )

        model = TDRPModel(cfg.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)

        best_rmse = float("inf")
        best_state = None
        for epoch in range(cfg.training.max_epochs):
            train_loss = _train_one_epoch(model, train_loader, optimizer, device)
            val_rmse, val_pearson = _evaluate(model, val_loader, device)
            history.append(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "val_pearson": val_pearson,
                }
            )
            logger.info(
                "Fold %d Epoch %d - train_loss=%.4f val_rmse=%.4f val_pearson=%.4f",
                fold_idx,
                epoch,
                train_loss,
                val_rmse,
                val_pearson,
            )
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = model.state_dict()

        if best_state is not None:
            torch.save(best_state, Path(output_dir) / f"model_fold{fold_idx}.pt")

    save_json(history, Path(output_dir) / "training_history.json")
    logger.info("Training complete. Models saved to %s", output_dir)
