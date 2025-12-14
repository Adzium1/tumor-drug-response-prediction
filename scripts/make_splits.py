"""
Generate filtered datasets and train/val/test splits for drug response modeling.

Steps:
- load processed labels/metadata
- filter drugs/cell lines by coverage
- emit random pair-wise, cell-line holdout, and tissue holdout splits

Example:
python scripts/make_splits.py \
  --config configs/default.yaml \
  --outdir data/processed/splits \
  --test-tissues lung_NSCLC urogenital_system leukemia aero_dig_tract breast
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.config import load_config
from tdrp.data.splits import (
    filter_entities,
    make_random_pair_split,
    make_cellline_holdout_split,
    make_tissue_holdout_split,
)
from tdrp.utils.io import load_parquet, ensure_dir
from tdrp.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build filtered datasets and splits.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to experiment config YAML.")
    parser.add_argument("--outdir", default="data/processed/splits", help="Where to save split CSVs.")
    parser.add_argument("--min-drug-frac", type=float, default=0.7, help="Min fraction of cell lines per drug.")
    parser.add_argument("--min-cell-frac", type=float, default=0.6, help="Min fraction of drugs per cell line.")
    parser.add_argument("--tissue-min-cells", type=int, default=15, help="Drop tissues with fewer cell lines than this.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--test-tissues",
        nargs="+",
        default=["lung_NSCLC", "urogenital_system", "leukemia", "aero_dig_tract", "breast"],
        help="Tissues to fully hold out for tissue-level evaluation.",
    )
    return parser.parse_args()


def _load_with_tissue(cfg, base: Path):
    labels_path = base / cfg.data.labels_file
    meta_path = base / (cfg.data.metadata_file or "")
    labels = load_parquet(labels_path)
    metadata = load_parquet(meta_path) if meta_path.exists() else None
    if metadata is None or "tissue" not in metadata.columns:
        raise ValueError("metadata.parquet with a 'tissue' column is required to build splits.")
    merged = labels.merge(metadata[["cell_line", "tissue"]], on="cell_line", how="left")
    return merged


def _save_split(df, path: Path, split_col: str) -> None:
    cols = [c for c in ["cell_line", "drug", "tissue", "ln_ic50", split_col] if c in df.columns]
    df[cols].to_csv(path, index=False)
    logging.info("Saved %s with %d rows", path, len(df))


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    base = Path(cfg.data.processed_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = _load_with_tissue(cfg, base)
    logging.info("Loaded %d pairs before filtering", len(df))

    filtered = filter_entities(
        df,
        min_drug_frac=args.min_drug_frac,
        min_cell_frac=args.min_cell_frac,
        tissue_min_cells=args.tissue_min_cells,
        tissue_col="tissue",
    )

    random_split = make_random_pair_split(filtered, seed=args.seed, split_col="split", stratify_by_tissue=True)
    # Explicit tissue stratification (default) preserves tissue proportions across splits
    # to mirror observed clustering in PCA.
    _save_split(random_split, outdir / "random_pair_split.csv", split_col="split")

    cell_holdout = make_cellline_holdout_split(
        filtered,
        tissue_col="tissue",
        seed=args.seed,
        split_col="split",
        min_tissue_cells=args.tissue_min_cells,
    )
    _save_split(cell_holdout, outdir / "cellline_holdout_split.csv", split_col="split")

    tissue_holdout = make_tissue_holdout_split(
        filtered,
        tissue_col="tissue",
        test_tissues=args.test_tissues,
        seed=args.seed,
        split_col="split",
    )
    _save_split(tissue_holdout, outdir / "tissue_holdout_split.csv", split_col="split")


if __name__ == "__main__":
    main()
