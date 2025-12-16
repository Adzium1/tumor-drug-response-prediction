"""
Plot a simple SHAP summary (mean |SHAP| bar chart) for the torch baseline.

Assumes SHAP values were saved by scripts/explain_torch_baseline.py and contain
`shap_values` (n_samples, n_features) and `feature_names`.

Example:
python scripts/plot_shap_baseline.py \
  --shap-file outputs/torch_baseline_random/shap_values.npz \
  --output outputs/torch_baseline_random/shap_top20.png \
  --top-k 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot mean |SHAP| bar chart for baseline SHAP values.")
    ap.add_argument("--shap-file", required=True, help="Path to shap_values.npz saved by explain_torch_baseline.py.")
    ap.add_argument("--output", required=True, help="Where to save the plot (e.g., outputs/.../shap_top20.png).")
    ap.add_argument("--top-k", type=int, default=20, help="Number of top features to display.")
    ap.add_argument("--title", default=None, help="Optional plot title.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data = np.load(args.shap_file, allow_pickle=True)
    shap_values = data["shap_values"]
    feature_names = data["feature_names"]

    # Handle potential object array (e.g., list of arrays) from SHAP APIs.
    if isinstance(shap_values, np.ndarray) and shap_values.dtype == object:
        shap_values = shap_values.item(0)

    if shap_values.ndim != 2:
        raise ValueError(f"Expected shap_values with ndim=2, got shape {shap_values.shape}")

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    top_k = min(args.top_k, len(mean_abs))
    idx = np.argsort(mean_abs)[-top_k:][::-1]

    top_features = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]
    top_importance = mean_abs[idx]

    plt.figure(figsize=(8, max(4, top_k * 0.3)))
    y_pos = np.arange(top_k)
    plt.barh(y_pos, top_importance, color="teal")
    plt.yticks(y_pos, top_features)
    plt.gca().invert_yaxis()
    plt.xlabel("Mean |SHAP value|")
    plt.title(args.title or f"Top {top_k} features by mean |SHAP|")
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=200)
    plt.close()
    logging.info("Saved SHAP plot to %s", args.output)


if __name__ == "__main__":
    main()
