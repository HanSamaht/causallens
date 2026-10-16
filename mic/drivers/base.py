"""驱动适配器基类：统一任意模型的"审讯接口"。

一个驱动器只需提供三件事：
  1. predict(X)          —— 前向传播（数值向量进，分数/概率出）
  2. background 数据      —— 模型训练分布的代表性样本（干预/反事实的取值参照）
  3. intervention 能力    —— 把某特征设为指定值再前向（因果图的发动机）

TorchDriver 额外提供内部层激活的读取与修补（activation patching）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class MicDriver:
    """所有驱动器的基类。"""

    kind = "base"
    prediction_semantics = "score"  # "probability" | "score"
    n_classes = 2                   # 2=二分类/回归单输出；>2 = 多分类
    target_class = 1                # 当前审讯的目标类别（归因/反事实围绕它展开）

    def __init__(
        self,
        model,
        feature_names: list[str],
        background: np.ndarray,
        name: str = "unnamed_model",
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.background = np.asarray(background, dtype=float)
        self.name = name
        assert self.background.shape[1] == len(self.feature_names), (
            f"background 列数 {self.background.shape[1]} 与特征数 {len(self.feature_names)} 不符"
        )

    # ---- 前向传播 -------------------------------------------------------
    def predict_full(self, X: np.ndarray) -> np.ndarray:
        """(n, C) 输出矩阵；默认实现=单输出包装（子类可覆写）。"""
        return self.predict(X).reshape(-1, 1)

    def predict_label(self, X: np.ndarray) -> np.ndarray:
        """argmax 类别标签（多分类语义）。"""
        return np.argmax(self.predict_full(X), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_one(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    # ---- 干预（intervention）：因果分析的原子操作 ------------------------
    def intervene(self, x: np.ndarray, assignments: dict[str, float]) -> np.ndarray:
        """do(feature := value)。返回干预后的新样本，不修改原样本。"""
        x2 = np.array(x, dtype=float, copy=True)
        for f, v in assignments.items():
            x2[self.feature_names.index(f)] = v
        return x2

    # ---- 背景分布工具 ----------------------------------------------------
    def feature_bounds(self, feature: str, lo: float = 0.005, hi: float = 0.995) -> tuple[float, float]:
        j = self.feature_names.index(feature)
        col = self.background[:, j]
        return float(np.quantile(col, lo)), float(np.quantile(col, hi))

    def quantile_of(self, feature: str, q: float) -> float:
        j = self.feature_names.index(feature)
        return float(np.quantile(self.background[:, j], q))

    def sample_at(self, idx: int) -> np.ndarray:
        return self.background[idx % len(self.background)].copy()

    # ---- 元信息 ----------------------------------------------------------
    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "prediction_semantics": self.prediction_semantics,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "n_background": len(self.background),
        }
