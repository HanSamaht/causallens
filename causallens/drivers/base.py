""":model"interrogationinterface".

driveronly:
  1. predict(X) —— (value,score/probability)
  2. background —— modeltrainingdistributionsample(intervention/counterfactualvalue)
  3. intervention —— featureset tovalue(causal graph)

TorchDriver layeractivationand(activation patching).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class MicDriver:
    """alldriver."""

    kind = "base"
    prediction_semantics = "score" # "probability" | "score"
    n_classes = 2 # 2=/output;>2 =
    target_class = 1 # wheninterrogation(attribution/counterfactual)

    def __init__(
        self,
        model,
        feature_names: list[str],
        background: np.ndarray,
        name: str = "unnamed_model",
    ):
        """Init."""
        self.model = model
        self.feature_names = list(feature_names)
        self.background = np.asarray(background, dtype=float)
        self.name = name
        assert self.background.shape[1] == len(self.feature_names), (
            f"background {self.background.shape[1]} andfeature {len(self.feature_names)} not"
        )

    # ---- -------------------------------------------------------
    def predict_full(self, X: np.ndarray) -> np.ndarray:
        """(n, C) output;default=output()."""
        return self.predict(X).reshape(-1, 1)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """argmax ()."""
        return np.argmax(self.predict_full(X), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict."""
        raise NotImplementedError

    def predict_one(self, x: np.ndarray) -> float:
        """Predict one."""
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    # ---- intervention(intervention):causalanalysis ------------------------
    def intervene(self, x: np.ndarray, assignments: dict[str, float]) -> np.ndarray:
        """do(feature := value).returninterventionsample,notmodifysample."""
        x2 = np.array(x, dtype=float, copy=True)
        for f, v in assignments.items():
            x2[self.feature_names.index(f)] = v
        return x2

    # ---- backgrounddistributiontool ----------------------------------------------------
    def feature_bounds(self, feature: str, lo: float = 0.005, hi: float = 0.995) -> tuple[float, float]:
        """Feature bounds."""
        j = self.feature_names.index(feature)
        col = self.background[:, j]
        return float(np.quantile(col, lo)), float(np.quantile(col, hi))

    def quantile_of(self, feature: str, q: float) -> float:
        """Quantile of."""
        j = self.feature_names.index(feature)
        return float(np.quantile(self.background[:, j], q))

    def sample_at(self, idx: int) -> np.ndarray:
        """Sample at."""
        return self.background[idx % len(self.background)].copy()

    # ---- ----------------------------------------------------------
    def describe(self) -> dict:
        """Describe."""
        return {
            "name": self.name,
            "kind": self.kind,
            "prediction_semantics": self.prediction_semantics,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "n_background": len(self.background),
        }
