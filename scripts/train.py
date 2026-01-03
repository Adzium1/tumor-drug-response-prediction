import argparse
import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.config import load_config
from tdrp.training.loop import train_model
from tdrp.utils.io import ensure_dir
from tdrp.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train TDRP model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--output", default="outputs", help="Output directory for models and logs.")
    parser.add_argument("--outdir", default=None, help="Alias for --output.")
    parser.add_argument("--split-csv", default=None, help="Optional split CSV to override split strategy.")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    if args.split_csv:
        cfg.data.split_csv = args.split_csv
    output_dir = args.outdir or args.output
    ensure_dir(output_dir)
    train_model(cfg, output_dir)


if __name__ == "__main__":
    main()
