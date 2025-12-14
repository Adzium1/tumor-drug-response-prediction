from __future__ import annotations

"""
Utilities to project learned embeddings to 2D for qualitative checks.

Use this to compare model embeddings against expression PCA: similar tissue
clustering indicates the network is capturing biological structure rather than
shortcuts.
"""

from pathlib import Path
from typing import Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

try:
    from umap import UMAP  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    UMAP = None

logger = logging.getLogger(__name__)


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def project_and_plot_embeddings(
    embeddings,
    metadata: pd.DataFrame,
    method: Literal["pca", "umap"] = "pca",
    color_by: str = "tissue",
    outpath: Optional[str | Path] = None,
    title: Optional[str] = None,
) -> tuple[np.ndarray, plt.Figure]:
    """
    Project embeddings to 2D (PCA or UMAP) and scatter color-coded by a metadata column.

    Parameters
    ----------
    embeddings : array-like (n_samples, d)
        Learned embeddings to visualize (NumPy or torch tensor).
    metadata : DataFrame
        Must align row-wise to embeddings; `color_by` column supplies labels (default: tissue).
    method : {"pca", "umap"}
        Dimensionality reduction method. UMAP is optional and will fall back to PCA if unavailable.
    color_by : str
        Column in metadata used to color points.
    outpath : str or Path, optional
        If provided, save the figure to this path.
    title : str, optional
        Title for the plot; defaults to method name + color_by.
    """
    emb = _to_numpy(embeddings)
    if emb.ndim != 2 or emb.shape[0] != len(metadata):
        raise ValueError("Embeddings must be 2D and match metadata length.")
    labels = metadata[color_by] if color_by in metadata.columns else pd.Series(["unknown"] * len(metadata))

    if method == "umap" and UMAP is not None:
        reducer = UMAP(n_components=2, random_state=0)
        coords = reducer.fit_transform(emb)
    else:
        # default PCA
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=0)
        coords = reducer.fit_transform(emb)
        if method == "umap" and UMAP is None:
            logger.warning("UMAP not installed; falling back to PCA.")

    plot_df = pd.DataFrame(coords, columns=["dim1", "dim2"])
    plot_df["label"] = labels.values

    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = plot_df["label"].fillna("unknown").astype(str)
    categories = pd.Categorical(unique_labels).categories
    cmap = plt.get_cmap("tab20")
    color_map = {cat: cmap(i / max(len(categories) - 1, 1)) for i, cat in enumerate(categories)}
    colors = unique_labels.map(color_map)
    ax.scatter(plot_df["dim1"], plot_df["dim2"], c=colors, s=18, alpha=0.8)
    # Build legend
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=color_map[c], label=c) for c in categories]
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title=color_by)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title or f"{method.upper()} colored by {color_by}")
    plt.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath, dpi=200)
        logger.info("Saved embedding plot to %s", outpath)

    return coords, fig
