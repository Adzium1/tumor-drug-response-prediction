"""
Project and visualize learned embeddings, colored by tissue (or another metadata column).

Example:
python scripts/plot_embeddings.py \
  --embeddings outputs/run1/cell_embeddings.npz \
  --metadata data/processed/metadata.parquet \
  --method pca \
  --color-by tissue \
  --outpath outputs/run1/embeddings_pca_tissue.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.analysis.embeddings import project_and_plot_embeddings
from tdrp.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot embeddings with PCA/UMAP colored by metadata.")
    parser.add_argument("--embeddings", required=True, help="Path to .npy or .npz containing embeddings.")
    parser.add_argument("--metadata", required=True, help="Path to metadata parquet file.")
    parser.add_argument("--method", default="pca", choices=["pca", "umap"], help="Projection method.")
    parser.add_argument("--color-by", default="tissue", help="Metadata column used for coloring.")
    parser.add_argument("--outpath", default="outputs/embeddings_pca_tissue.png", help="Where to save the plot.")
    return parser.parse_args()


def _load_embeddings(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "embeddings" in arr:
            return arr["embeddings"]
        # fallback: first array
        first_key = list(arr.keys())[0]
        return arr[first_key]
    return np.asarray(arr)


def main() -> None:
    setup_logging()
    args = parse_args()
    emb_path = Path(args.embeddings)
    meta_path = Path(args.metadata)
    metadata = pd.read_parquet(meta_path)
    embeddings = _load_embeddings(emb_path)
    project_and_plot_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        method=args.method,
        color_by=args.color_by,
        outpath=args.outpath,
    )


if __name__ == "__main__":
    main()
