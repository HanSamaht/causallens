"""时序驱动器：让滑动窗口模型（量化收益率预测、负荷预测……）接入麦克风。

核心设计：内部采用"展平表示"。背景数据 (n, T, F) 被存为 (n, T*F)，
特征名为 `{feature}@lag{k}`（lag0 = 窗口内最新一步）。这样归因、干预、
反事实、因果图、忠实性——整套分析管线零改动复用；前向时重塑回 (B, T, F)。

继承 TorchDriver：免费获得激活钩子、激活修补（含序列双语义单元修补）、
steering、梯度归因——只覆写输入张量构造。

干预语义：
  - intervene({"momentum": v})       —— 全窗口干预：整个窗口的 momentum 都设为 v
  - intervene({"momentum@lag0": v})   —— 单点干预：只改最新一步
  - intervene_window(lag_slice, ...)  —— 只干预某个历史区段
"""
from __future__ import annotations

import numpy as np

from .torch_model import TorchDriver


def make_flat_names(feature_names: list[str], seq_len: int) -> list[str]:
    return [f"{f}@lag{seq_len - 1 - t}" for t in range(seq_len) for f in feature_names]


class TimeSeriesDriver(TorchDriver):
    kind = "timeseries"
    prediction_semantics = "score"  # 默认回归（如收益率预测）；分类时可覆盖

    def __init__(self, model, feature_names, background, name="ts_model", seq_len: int | None = None):
        bg = np.asarray(background, dtype=float)
        if bg.ndim == 3:
            seq_len = seq_len or bg.shape[1]
            self.n_features_raw = bg.shape[2]
            bg = bg.reshape(len(bg), -1)
        else:
            assert seq_len is not None, "背景为 2D 时必须显式给 seq_len"
            self.n_features_raw = len(feature_names)
        self.seq_len = seq_len
        self.base_feature_names = list(feature_names)
        try:
            import torch

            self.torch = torch
        except ImportError as e:
            raise ImportError("TimeSeriesDriver 需要 torch：pip install torch") from e
        super().__init__(model, make_flat_names(feature_names, seq_len), bg, name=name)

    # ---- 输入构造：flat → (B, T, F) ------------------------------------
    def _to_tensor(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = len(X)
        return self.torch.tensor(
            X.reshape(n, self.seq_len, self.n_features_raw), dtype=self.torch.float32
        )

    # ---- 时序专属干预 -----------------------------------------------------
    def intervene(self, x: np.ndarray, assignments: dict[str, float]) -> np.ndarray:
        """基础特征名（不带 @lag）→ 全窗口干预；带 @lag 的名 → 单点干预。"""
        expanded: dict[str, float] = {}
        for f, v in assignments.items():
            if "@" in f:
                expanded[f] = v
            else:
                for lag in range(self.seq_len):
                    expanded[f"{f}@lag{lag}"] = v
        return super().intervene(x, expanded)

    def intervene_window(self, x: np.ndarray, lag_from: int, lag_to: int, assignments: dict[str, float]) -> np.ndarray:
        """只干预 [lag_from, lag_to] 区段（含端点；lag0=最新）。"""
        expanded = {}
        for f, v in assignments.items():
            for lag in range(lag_from, lag_to + 1):
                expanded[f"{f}@lag{lag}"] = v
        return self.intervene(x, expanded)

    def window_attribution(self, x: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """梯度×输入 → (T, F) 归因热力图 + 平铺特征名。"""
        t = self.torch
        x_flat = np.asarray(x, dtype=float).ravel()
        xt = t.tensor(x_flat.reshape(1, self.seq_len, self.n_features_raw), dtype=t.float32)
        xt.requires_grad_(True)
        out = self.model(xt).squeeze()
        out.backward()
        grad = xt.grad.detach().cpu().numpy()
        heat = grad[0] * xt.detach().cpu().numpy()[0]
        return heat, self.feature_names

    # ---- 元信息 ------------------------------------------------------------
    def describe(self) -> dict:
        d = super().describe()
        d["seq_len"] = self.seq_len
        d["base_features"] = self.base_feature_names
        d["note"] = "特征名为 {feature}@lag{k}，lag0=窗口最新一步；激活修补含 h_T/持久单元双语义"
        return d
