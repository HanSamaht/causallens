"""FastSHAP 摊销解释器：全分辨率 SHAP 的毫秒级替代。

文献依据：FastSHAP (Jethani et al., ICLR 2022, 337 引)——训练一个神经网络
来预测 SHAP 值，摊销后一次前向出结果，比采样法快几个数量级。

对 MIC 时序模型的意义：`full` 档（T×F 个特征）从分钟级变毫秒级。
"""
from __future__ import annotations

import numpy as np


class AmortizedExplainer:
    """轻量摊销解释器：小 MLP 学习 (输入→SHAP值) 的映射。

    训练一次后，后续任意样本的 SHAP 归因只需一次前向传播。
    模型很小（几百参数），CPU 秒级训练。
    """

    def __init__(self, n_features: int, hidden: int = 128, seed: int = 0):
        import torch

        self.torch = torch
        self.torch.manual_seed(seed)
        self.n_features = n_features
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_features),  # 输出：每个特征的 SHAP 值
        )
        self.fitted = False

    def fit(self, get_shap_fn, background: np.ndarray, n_samples: int = 200,
            epochs: int = 200, lr: float = 1e-3, verbose: bool = False) -> dict:
        """训练摊销解释器。

        get_shap_fn: 函数，输入 (n, F) 矩阵，返回 (n, F) SHAP 矩阵
        background:  背景数据 (n, F)
        """
        t = self.torch
        rng = np.random.default_rng(0)
        idx = rng.choice(len(background), size=min(n_samples, len(background)), replace=False)
        X_sub = background[idx]

        # 用真实 SHAP 做训练目标
        shap_targets = np.array([get_shap_fn(X_sub[i:i+1])[0] for i in range(len(X_sub))])

        X_t = t.tensor(X_sub, dtype=t.float32)
        Y_t = t.tensor(shap_targets, dtype=t.float32)

        opt = t.optim.Adam(self.net.parameters(), lr=lr)
        lossf = t.nn.MSELoss()
        for ep in range(epochs):
            opt.zero_grad()
            pred = self.net(X_t)
            loss = lossf(pred, Y_t)
            loss.backward()
            opt.step()
            if verbose and (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1}: loss={loss.item():.6f}")

        with t.no_grad():
            final_loss = float(lossf(self.net(X_t), Y_t))
        self.fitted = True
        return {"n_training_samples": len(X_sub), "final_mse": round(final_loss, 6), "epochs": epochs}

    def attribute(self, x: np.ndarray) -> np.ndarray:
        """一次前向传播，返回 SHAP 近似值。"""
        assert self.fitted, "先调用 fit()"
        t = self.torch
        with t.no_grad():
            x_t = t.tensor(np.asarray(x, dtype=np.float32).reshape(1, -1), dtype=t.float32)
            return self.net(x_t).cpu().numpy()[0]
