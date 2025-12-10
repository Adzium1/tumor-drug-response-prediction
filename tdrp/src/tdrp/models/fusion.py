from __future__ import annotations

from typing import List
import torch
from torch import nn

from tdrp.config import ModelConfig
from tdrp.models.vae import OmicsVAE, vae_loss
from tdrp.models.encoders import OmicsEncoder, DrugEncoder


class FusionRegressor(nn.Module):
    def __init__(self, omics_latent_dim: int, drug_latent_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = omics_latent_dim + drug_latent_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, omics_latent: torch.Tensor, drug_latent: torch.Tensor) -> torch.Tensor:
        x = torch.cat([omics_latent, drug_latent], dim=-1)
        return self.net(x).squeeze(-1)


class TDRPModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.use_vae = cfg.use_vae
        self.omics_dim = cfg.omics_dim
        self.drug_dim = cfg.drug_dim
        self.lambda_vae = 0.1

        if self.use_vae:
            self.vae = OmicsVAE(input_dim=cfg.omics_dim, latent_dim=cfg.omics_latent_dim)
            omics_encoder_input_dim = cfg.omics_dim
        else:
            self.vae = None
            omics_encoder_input_dim = cfg.omics_dim

        self.omics_encoder = OmicsEncoder(
            input_dim=omics_encoder_input_dim,
            latent_dim=cfg.omics_latent_dim,
            dropout=cfg.dropout,
        )
        self.drug_encoder = DrugEncoder(
            input_dim=cfg.drug_dim,
            latent_dim=cfg.omics_latent_dim,
            dropout=cfg.dropout,
        )
        self.fusion = FusionRegressor(
            omics_latent_dim=cfg.omics_latent_dim,
            drug_latent_dim=cfg.omics_latent_dim,
            hidden_dims=cfg.fusion_hidden_dims,
            dropout=cfg.dropout,
        )

    def forward(self, omics: torch.Tensor, drug_fp: torch.Tensor):
        if self.use_vae and self.vae is not None:
            recon, mu, logvar = self.vae(omics)
            omics_input = recon.detach()
        else:
            mu = logvar = None
            omics_input = omics

        omics_latent = self.omics_encoder(omics_input)
        drug_latent = self.drug_encoder(drug_fp)
        preds = self.fusion(omics_latent, drug_latent)
        return preds, mu, logvar
