"""回测引擎（最小但忠实）：棘轮止损 + 双成交语义。

复刻真实案例的病灶机制——同一策略、同一数据，只差一个成交假设：
  fill_market = 0（阶梯价/乐观）：止损触发时假设能按止损线价格成交（回测器的错）
  fill_market = 1（市价/现实）：止损触发时价格早已跌穿，按当根 K 线收盘价成交

语义细节（防自身引入前视偏差）：
  - 棘轮止损位由【之前】bars 的峰值计算，先查触发、后更新峰值
  - 同 bar 同时触及止损与止盈时按止损优先（保守）
  - 入场在 entry_every 整数倍的 bar 开盘价；费用与滑点双边收取、方向不利
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BTParams:
    fill_market: float = 1.0   # 0=阶梯价(乐观) 1=市价(现实)；0.5=混合（取两种成交价均值）
    stop_pct: float = 0.03     # 棘轮止损距离（峰值 - stop_pct）
    tp_pct: float = 0.12       # 止盈距离
    fee_bps: float = 2.0       # 单边费率（bps）
    slippage_bps: float = 0.0  # 额外不利滑点（bps，压力测试用）
    entry_every: float = 24.0  # 每 N 根 bar 入场一次


def run_backtest(ohlc: np.ndarray, params: BTParams) -> dict:
    """ohlc: (T, 4) 开高低收。返回逐笔记录与汇总指标。"""
    o, h, l, c = (ohlc[:, k] for k in range(4))
    cost = (params.fee_bps + params.slippage_bps) / 1e4
    entry_every = max(1, int(round(params.entry_every)))

    trades: list[dict] = []
    in_pos = False
    entry = peak = stop = tp = 0.0

    for i in range(1, len(c)):
        if not in_pos and (i % entry_every == 0):
            entry = o[i] * (1 + cost)          # 买入承担不利成本
            peak, tp = entry, entry * (1 + params.tp_pct)
            in_pos = True
            continue

        if not in_pos:
            continue

        stop = peak * (1 - params.stop_pct)    # 峰值来自之前的 bars（防前视）
        exit_price, reason = None, None
        if l[i] <= stop:                        # 止损触发
            if params.fill_market < 0.5:        # 阶梯价：按止损线记账（乐观假设）
                exit_price, reason = stop, "stop_ladder"
            elif params.fill_market > 0.5:      # 市价：按当 bar 收盘（已跌穿）
                exit_price, reason = min(stop, c[i]), "stop_market"
            else:
                exit_price = 0.5 * (stop + min(stop, c[i]))
                reason = "stop_blend"
        elif h[i] >= tp:                        # 止盈：限价单按线成交（合理）
            exit_price, reason = tp, "tp"

        if exit_price is not None:
            exit_price *= (1 - cost)            # 卖出承担不利成本
            trades.append(
                {
                    "entry_bar": i,
                    "pnl_pct": exit_price / entry - 1,
                    "reason": reason,
                    "gap_through": max(0.0, (stop - min(stop, c[i])) / stop) if reason != "tp" else 0.0,
                }
            )
            in_pos = False
        else:
            peak = max(peak, h[i])              # 触发检查之后才更新峰值

    pnls = np.array([t["pnl_pct"] for t in trades]) if trades else np.array([0.0])
    wins, losses = pnls[pnls > 0], pnls[pnls <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 100.0
    return {
        "n_trades": len(trades),
        "win_rate": float((pnls > 0).mean()),
        "pf": float(min(pf, 100.0)),
        "avg_pnl_pct": float(pnls.mean()),
        "n_stop": sum(1 for t in trades if t["reason"] != "tp"),
        "avg_gap_through": float(np.mean([t["gap_through"] for t in trades])) if trades else 0.0,
        "trades": trades,
    }


def make_synthetic_4h(n_bars: int = 6000, seed: int = 7) -> np.ndarray:
    """合成 4H OHLC：趋势性长跑 + 间歇性暴力跳空 bar（复刻真实病灶结构）。

    结构要点：温和趋势使棘轮峰值远高于入场价；偶发跳空跌穿止损线——
    阶梯价假设把这些出场记成"峰值-止损距"的盈利，市价则按已跌穿的收盘价记账。
    """
    rng = np.random.default_rng(seed)
    # AR(1) 趋势：正自相关制造长跑；跳空制造跌穿事件
    eps = rng.normal(0.0, 0.0055, size=n_bars)
    ret = np.empty(n_bars)
    ret[0] = 0.0018 + eps[0]
    for t in range(1, n_bars):
        ret[t] = 0.0018 + 0.22 * ret[t - 1] + eps[t]
    jump = rng.random(n_bars) < 0.030                      # 3% 的 bar 暴力跳空
    ret[jump] -= rng.uniform(0.07, 0.11, size=jump.sum())
    close = 100.0 * np.cumprod(1 + ret)
    open_ = np.empty_like(close)
    open_[0], open_[1:] = close[0], close[:-1]
    hi = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.002, n_bars))
    lo = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.002, n_bars))
    return np.column_stack([open_, hi, lo, close])
