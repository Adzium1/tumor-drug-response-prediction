from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from tdrp.data.drug_enrichment import enrich_drugs


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich GDSC drug metadata via PubChem/ChEMBL.")
    parser.add_argument("--processed-dir", default="data/processed", help="Processed data directory.")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Optional directory for per-drug Markdown reports.",
    )
    parser.add_argument("--max-drugs", type=int, default=None, help="Optional cap for number of drugs.")
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional sleep between API calls to avoid rate limiting.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    enrich_drugs(
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        max_drugs=args.max_drugs,
        sleep_sec=args.sleep_sec,
    )


if __name__ == "__main__":
    main()
