from dataclasses import dataclass, field
from typing import Optional, Any
import yaml


@dataclass
class DataConfig:
    processed_dir: str
    omics_file: str = "omics.parquet"
    drug_file: str = "drug_fingerprints.parquet"
    labels_file: str = "labels.parquet"
    metadata_file: Optional[str] = "metadata.parquet"
    n_genes: int = 2000
    fingerprint_bits: int = 1024


@dataclass
class ModelConfig:
    use_vae: bool = False
    omics_dim: int = 2000
    omics_latent_dim: int = 128
    drug_dim: int = 1024
    fusion_hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    split_strategy: str = "leave_cell_line_out"
    k_folds: int = 5
    tissue_holdout: Optional[str] = None
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _dict_to_dataclass(cls: Any, data: dict):
    return cls(**data)


def load_config(path: str) -> ExperimentConfig:
    """Load YAML config from path into ExperimentConfig dataclasses."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration file must contain a mapping at the top level.")
    data_cfg = _dict_to_dataclass(DataConfig, raw.get("data", {}))
    model_cfg = _dict_to_dataclass(ModelConfig, raw.get("model", {}))
    training_cfg = _dict_to_dataclass(TrainingConfig, raw.get("training", {}))
    return ExperimentConfig(data=data_cfg, model=model_cfg, training=training_cfg)


def config_to_dict(cfg: ExperimentConfig) -> dict:
    """Convert ExperimentConfig to a serializable dict."""
    return {
        "data": cfg.data.__dict__,
        "model": cfg.model.__dict__,
        "training": cfg.training.__dict__,
    }
