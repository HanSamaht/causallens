"""modeldriver: Booster IC LightGBM/XGBoost .

(v3d_prod:3 × 3 × 9 Booster).
Prediction and SHAP stay consistent by construction:
output = sum of w_i times f_i(x) and SHAP = sum of w_i times SHAP_i(x),
"""
from __future__ import annotations

import numpy as np

from .tree import TreeDriver


class _EnsembleExplainer:
    """shap.TreeExplainer ( attribution layer)."""

    def __init__(self, members: list[tuple], weights: np.ndarray):
        """Init."""
        import shap
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*shap values output has changed.*")
            self.explainers = [shap.TreeExplainer(m[0]) for m in members]
        self.weights = weights
        base = 0.0
        for w, ex in zip(weights, self.explainers):
            try:
                base += w * float(np.ravel(ex.expected_value)[0])
            except Exception:
                pass
        self.expected_value = base

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Shap values."""
        X = np.asarray(X, dtype=float)
        out = None
        for w, ex in zip(self.weights, self.explainers):
            sv = np.asarray(ex.shap_values(X), dtype=float)
            out = w * sv if out is None else out + w * sv
        return out


class EnsembleTreeDriver(TreeDriver):
    """Ensembletreedriver."""
    kind = "ensemble-tree"

    def __init__(self, members: list, feature_names, background, name="ensemble"):
        """members: [(model, weight, ...), ...] or a plain model list;
        both the production v3d_prod and the book models bundles load."""
        self.members = [m if isinstance(m, (tuple, list)) else (m, 1.0) for m in members]
        w = np.array([float(m[1]) for m in self.members], dtype=float)
        self.weights = w / w.sum()
        super().__init__(self.members, feature_names, background, name=name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        X = np.asarray(X, dtype=float)
        out = None
        for w, m in zip(self.weights, self.members):
            p = m[0].predict(X)
            out = w * p if out is None else out + w * p
        return np.asarray(out, dtype=float)
