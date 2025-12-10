from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class OmicsPCA:
    """Wrapper around sklearn PCA for gene expression."""

    def __init__(self, n_components: int):
        self.n_components = n_components
        self.model = PCA(n_components=n_components)

    def fit(self, X: np.ndarray) -> "OmicsPCA":
        self.model.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.fit_transform(X)

    @property
    def explained_variance_ratio_(self):
        return getattr(self.model, "explained_variance_ratio_", None)
