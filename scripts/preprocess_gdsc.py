import argparse
from pathlib import Path

from tdrp.data.gdsc2_preprocess import preprocess_gdsc2
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
    print("Saved omics:", (processed_path / "omics.parquet").resolve())
    print("Saved drug_fingerprints:", (processed_path / "drug_fingerprints.parquet").resolve())
    print("Saved labels:", (processed_path / "labels.parquet").resolve())
    print("Saved metadata:", (processed_path / "metadata.parquet").resolve())


if __name__ == "__main__":
    main()
