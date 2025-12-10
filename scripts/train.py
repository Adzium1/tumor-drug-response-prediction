import argparse

from tdrp.config import load_config
from tdrp.training.loop import train_model
from tdrp.utils.io import ensure_dir
from tdrp.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train TDRP model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML.")
    parser.add_argument("--output", default="outputs", help="Output directory for models and logs.")
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dir(args.output)
    train_model(cfg, args.output)


if __name__ == "__main__":
    main()
