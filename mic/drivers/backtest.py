"""BacktestDriver：审讯对象是回测引擎（评估器），不是 ML 模型。

"特征" = 执行参数与假设（成交语义/止损距/止盈距/费率/滑点/入场频率）
"预测" = 盈利因子 PF（score 语义，盈利线 1.0 是反事实翻转阈值）

于是整套审讯管线自动获得新含义：
  - 归因：     哪些执行假设在驱动回测数字（SHAP over 执行参数）
  - 干预：     do(成交语义 := 市价) —— C'→C 实验成为一等公民
  - 反事实：   哪个最小参数改动让 PF 跌破盈利线
  - 因果图：   执行参数间的交互（如 止损距 × 成交语义）
  - 忠实性：   归因排名 vs 干预实测敏感度（解释配不配得上引擎行为）
"""
from __future__ import annotations

import numpy as np

from ..backtest.engine import BTParams, run_backtest
from .base import MicDriver

PARAM_NAMES = ["fill_market", "stop_pct", "tp_pct", "fee_bps", "slippage_bps", "entry_every"]


class BacktestDriver(MicDriver):
    kind = "backtest"
    prediction_semantics = "score"  # PF：1.0 = 盈亏平衡线

    def __init__(self, ohlc: np.ndarray, feature_names=None, background=None, name="backtest_engine"):
        self.ohlc = np.asarray(ohlc, dtype=float)
        feature_names = list(feature_names or PARAM_NAMES)
        if background is None:
            # 背景网格：围绕基线参数的代表性执行假设组合（含两侧成交语义）
            rng = np.random.default_rng(0)
            n = 40
            bg = np.column_stack(
                [
                    rng.choice([0.0, 1.0], size=n),                  # 成交语义
                    rng.uniform(0.02, 0.06, size=n),                 # 止损距
                    rng.uniform(0.08, 0.20, size=n),                 # 止盈距
                    rng.uniform(0.0, 8.0, size=n),                   # 费率 bps
                    rng.uniform(0.0, 30.0, size=n),                  # 滑点 bps
                    rng.choice([6, 12, 24, 48], size=n).astype(float),  # 入场频率
                ]
            )
            bg[:2] = [0.0, 0.03, 0.15, 2.0, 0.0, 12.0]   # 基线：乐观阶梯
            bg[2:4] = [1.0, 0.03, 0.15, 2.0, 0.0, 12.0]  # 基线：现实市价
            background = bg
        super().__init__(self.ohlc, feature_names, background, name=name)

    # ---- "前向传播"：参数 → PF -------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.empty(len(X))
        for i, row in enumerate(X):
            params = BTParams(**dict(zip(PARAM_NAMES, row)))
            out[i] = run_backtest(self.ohlc, params)["pf"]
        return out

    def run(self, **params) -> dict:
        """按命名参数跑一次完整回测，返回全部指标（含逐笔）。"""
        base = dict(zip(PARAM_NAMES, self.background[0]))
        merged = {**{k: base[k] for k in PARAM_NAMES}, **params}
        return run_backtest(self.ohlc, BTParams(**merged))

    def describe(self) -> dict:
        d = super().describe()
        d["note"] = (
            "审讯对象是回测引擎：feature=执行参数假设，prediction=盈利因子PF（1.0为盈亏平衡）。"
            "fill_market: 0=阶梯价记账(乐观) 1=市价成交(现实)"
        )
        d["n_bars"] = int(len(self.ohlc))
        return d
