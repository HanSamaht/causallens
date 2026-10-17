"""modeldriver:sklearn / LightGBM / XGBoost.

support//: n_classes>2,interrogation target_class
(session.set_target_class(k) switch),counterfactualas"argmax ".
"""
from __future__ import annotations

import numpy as np

from .base import MicDriver


class TreeDriver(MicDriver):
    kind = "tree"

    def __init__(self, model, feature_names, background, name="tree_model", semantics=None):
        self.model = model
        # : predict_proba width; proba -> output
        if hasattr(model, "predict_proba"):
            n_feat_out = model.predict_proba(np.asarray(background, dtype=float)[:2]).shape[1]
            self.n_classes = max(2, n_feat_out)
            self.prediction_semantics = semantics or "probability"
        else:
            self.n_classes = 1
            self.prediction_semantics = semantics or "score"
        super().__init__(model, feature_names, background, name)

    def predict_full(self, X: np.ndarray) -> np.ndarray:
        """(n, C) probability;return (n, 1) score."""
        X = np.asarray(X, dtype=float)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return self.model.predict(X).reshape(-1, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        full = self.predict_full(X)
        if full.shape[1] == 1:
            return full[:, 0]
        return full[:, min(self.target_class, full.shape[1] - 1)]

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """argmax (interrogation)."""
        return np.argmax(self.predict_full(X), axis=1)
