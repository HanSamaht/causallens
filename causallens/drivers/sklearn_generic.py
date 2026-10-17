""" sklearn driver: predict / predict_proba model(SVM,KNN,MLP……).

model——attribution/Kernel ,intervention.
"""
from __future__ import annotations

import numpy as np

from .base import MicDriver


class GenericSklearnDriver(MicDriver):
    """Genericsklearndriver."""
    kind = "sklearn"

    def __init__(self, model, feature_names, background, name="sklearn_model", semantics=None):
        """Init."""
        self.model = model
        if hasattr(model, "predict_proba"):
            n_out = model.predict_proba(np.asarray(background, dtype=float)[:2]).shape[1]
            self.n_classes = max(2, n_out)
            self.prediction_semantics = semantics or "probability"
        else:
            self.n_classes = 1
            self.prediction_semantics = semantics or "score"
        super().__init__(model, feature_names, background, name)

    def predict_full(self, X: np.ndarray) -> np.ndarray:
        """Predict full."""
        X = np.asarray(X, dtype=float)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return self.model.predict(X).reshape(-1, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        full = self.predict_full(X)
        if full.shape[1] == 1:
            return full[:, 0]
        return full[:, min(self.target_class, full.shape[1] - 1)]

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """Predict label."""
        return np.argmax(self.predict_full(X), axis=1)
