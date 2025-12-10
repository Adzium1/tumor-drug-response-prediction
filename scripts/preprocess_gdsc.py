import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.data.gdsc2_preprocess import preprocess_gdsc2
from tdrp.utils.io import load_parquet
from tdrp.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GDSC2 raw files into parquet tables.")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw GDSC2 files.")
    parser.add_argument("--outdir", default="data/processed", help="Output directory for processed parquet files.")
    parser.add_argument("--n-genes", type=int, default=2000, help="Number of top-variance genes to keep.")
    parser.add_argument("--fingerprint-bits", type=int, default=1024, help="Number of bits for Morgan fingerprints.")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    preprocess_gdsc2(
        raw_dir=args.raw_dir,
        processed_dir=args.outdir,
        n_genes=args.n_genes,
        fingerprint_bits=args.fingerprint_bits,
    )
    processed_path = Path(args.outdir)
    omics_df = load_parquet(processed_path / "omics.parquet")
    drug_fp_df = load_parquet(processed_path / "drug_fingerprints.parquet")
    labels_df = load_parquet(processed_path / "labels.parquet")
    metadata_df = load_parquet(processed_path / "metadata.parquet")
    print("Saved omics:", (processed_path / "omics.parquet").resolve(), "shape", omics_df.shape)
    print("Saved drug_fingerprints:", (processed_path / "drug_fingerprints.parquet").resolve(), "shape", drug_fp_df.shape)
    print("Saved labels:", (processed_path / "labels.parquet").resolve(), "shape", labels_df.shape)
    print("Saved metadata:", (processed_path / "metadata.parquet").resolve(), "shape", metadata_df.shape)


if __name__ == "__main__":
    main()
