import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.config import load_config
from tdrp.models.fusion import TDRPModel
from tdrp.analysis.shap_analysis import compute_shap_values
from tdrp.utils.io import load_parquet
from tdrp.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SHAP values for TDRP model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--output", default="outputs/shap_values.npz", help="Output .npz file path.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples to explain.")
    parser.add_argument("--background-size", type=int, default=100, help="Background sample size for SHAP.")
    parser.add_argument("--device", default="cpu", help="Device for model evaluation.")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    omics_df = load_parquet(Path(cfg.data.processed_dir) / cfg.data.omics_file)
    drug_df = load_parquet(Path(cfg.data.processed_dir) / cfg.data.drug_file)
    labels_df = load_parquet(Path(cfg.data.processed_dir) / cfg.data.labels_file)

    cfg.model.omics_dim = omics_df.shape[1] - 1
    cfg.model.drug_dim = drug_df.shape[1] - 1

    device = torch.device(args.device)
    model = TDRPModel(cfg.model).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    results = compute_shap_values(
        model=model,
        omics_df=omics_df,
        drug_df=drug_df,
        labels_df=labels_df,
        sample_size=args.sample_size,
        background_size=args.background_size,
        device=args.device,
    )
    np.savez(args.output, **results)
    logging.info("Saved SHAP values to %s", args.output)


if __name__ == "__main__":
    main()
