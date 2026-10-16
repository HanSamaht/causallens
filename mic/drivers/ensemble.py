"""集成树模型驱动器：多 Booster 按 IC 加权的 LightGBM/XGBoost 集成。

真实量化场景的标准形态（v3d_prod：3 跨度 × 3 种子 × 9 个 Booster）。
预测与 SHAP 都可精确分解：集成输出 = Σ wᵢ·fᵢ(x)，集成 SHAP = Σ wᵢ·SHAPᵢ(x)，
基准值同理——加法性使然，无近似。
"""
from __future__ import annotations

import numpy as np

from .tree import TreeDriver


class _EnsembleExplainer:
    """shap.TreeExplainer 的加权聚合适配器（供 attribution 层调用）。"""

    def __init__(self, members: list[tuple], weights: np.ndarray):
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
        X = np.asarray(X, dtype=float)
        out = None
        for w, ex in zip(self.weights, self.explainers):
            sv = np.asarray(ex.shap_values(X), dtype=float)
            out = w * sv if out is None else out + w * sv
        return out


class EnsembleTreeDriver(TreeDriver):
    kind = "ensemble-tree"

    def __init__(self, members: list, feature_names, background, name="ensemble"):
        """members: [(模型, 权重, ...), ...] 或 [模型, ...]（等权）——v3d_prod 与 book_models 两种格式都收。"""
        self.members = [m if isinstance(m, (tuple, list)) else (m, 1.0) for m in members]
        w = np.array([float(m[1]) for m in self.members], dtype=float)
        self.weights = w / w.sum()
        super().__init__(self.members, feature_names, background, name=name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        out = None
        for w, m in zip(self.weights, self.members):
            p = m[0].predict(X)
            out = w * p if out is None else out + w * p
        return np.asarray(out, dtype=float)
