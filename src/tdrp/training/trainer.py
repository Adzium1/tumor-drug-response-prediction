from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from tdrp.training.metrics import rmse, pearsonr


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_rmse: float
    val_pearson: float


class Trainer:
    """Lightweight trainer with early stopping on validation RMSE."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()
            preds = self.model(batch["x"])
            loss = self.criterion(preds, batch["y"])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(batch["y"])
            total_samples += len(batch["y"])
        return total_loss / max(total_samples, 1)

    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                preds = self.model(batch["x"])
                preds_list.append(preds.detach().cpu())
                targets_list.append(batch["y"].detach().cpu())
        if not preds_list:
            return {"rmse": float("nan"), "pearson": float("nan")}
        y_pred = torch.cat(preds_list).numpy()
        y_true = torch.cat(targets_list).numpy()
        return {"rmse": rmse(y_true, y_pred), "pearson": pearsonr(y_true, y_pred)}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 10,
        min_delta: float = 0.0,
    ) -> List[EpochResult]:
        best_state: Optional[dict] = None
        best_rmse = float("inf")
        epochs_no_improve = 0
        history: List[EpochResult] = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            if self.scheduler:
                self.scheduler.step(val_metrics["rmse"])
            history.append(
                EpochResult(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_rmse=val_metrics["rmse"],
                    val_pearson=val_metrics["pearson"],
                )
            )
            improved = val_metrics["rmse"] + min_delta < best_rmse
            if improved:
                best_rmse = val_metrics["rmse"]
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history
