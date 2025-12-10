import argparse
import logging

from tdrp.data.gdsc_loader import preprocess_gdsc
from tdrp.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess GDSC-style datasets for TDRP.")
    parser.add_argument("--expression", required=True, help="Path to expression CSV.")
    parser.add_argument("--labels", required=True, help="Path to labels CSV.")
    parser.add_argument("--drug-smiles", required=True, help="Path to drug SMILES CSV.")
    parser.add_argument("--metadata", default=None, help="Optional metadata CSV path.")
    parser.add_argument("--outdir", default="tdrp/data/processed", help="Output directory.")
    parser.add_argument("--n-genes", type=int, default=2000, help="Number of top-variance genes.")
    parser.add_argument("--fingerprint-bits", type=int, default=1024, help="Fingerprint bits.")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    preprocess_gdsc(
        expression_path=args.expression,
        labels_path=args.labels,
        drug_smiles_path=args.drug_smiles,
        metadata_path=args.metadata,
        outdir=args.outdir,
        n_genes=args.n_genes,
        fingerprint_bits=args.fingerprint_bits,
    )


if __name__ == "__main__":
    main()
