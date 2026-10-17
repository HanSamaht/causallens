"""backtestengine(most): + fill semantics.

——,,onlyhypothesis:
  fill_market = 0(/):hypothesis(backtest)
  fill_market = 1(/):,when K

(bias):
  - []bars valuecompute,,updatevalue
  - bar andand()
  - in entry_every bar ;and,directionnot
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BTParams:
    """Btparams."""
    fill_market: float = 1.0 # 0=() 1=();0.5=(value)
    stop_pct: float = 0.03 # (value - stop_pct)
    tp_pct: float = 0.12 #
    fee_bps: float = 2.0 # (bps)
    slippage_bps: float = 0.0 # not(bps,test)
    entry_every: float = 24.0 # every N bar


def run_backtest(ohlc: np.ndarray, params: BTParams) -> dict:
    """ohlc: (T, 4) .returnrecordand."""
    o, h, l, c = (ohlc[:, k] for k in range(4))
    cost = (params.fee_bps + params.slippage_bps) / 1e4
    entry_every = max(1, int(round(params.entry_every)))

    trades: list[dict] = []
    in_pos = False
    entry = peak = stop = tp = 0.0

    for i in range(1, len(c)):
        if not in_pos and (i % entry_every == 0):
            entry = o[i] * (1 + cost) # not
            peak, tp = entry, entry * (1 + params.tp_pct)
            in_pos = True
            continue

        if not in_pos:
            continue

        stop = peak * (1 - params.stop_pct) # value bars()
        exit_price, reason = None, None
        if l[i] <= stop: #
            if params.fill_market < 0.5: # :(hypothesis)
                exit_price, reason = stop, "stop_ladder"
            elif params.fill_market > 0.5: # :when bar ()
                exit_price, reason = min(stop, c[i]), "stop_market"
            else:
                exit_price = 0.5 * (stop + min(stop, c[i]))
                reason = "stop_blend"
        elif h[i] >= tp: # :()
            exit_price, reason = tp, "tp"

        if exit_price is not None:
            exit_price *= (1 - cost) # not
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
            peak = max(peak, h[i]) # checkupdatevalue

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
    """ 4H OHLC: + bar().

    :andvalue;——
    hypothesisthis"value-",.
    """
    rng = np.random.default_rng(seed)
    # AR(1) :correlation;
    eps = rng.normal(0.0, 0.0055, size=n_bars)
    ret = np.empty(n_bars)
    ret[0] = 0.0018 + eps[0]
    for t in range(1, n_bars):
        ret[t] = 0.0018 + 0.22 * ret[t - 1] + eps[t]
    jump = rng.random(n_bars) < 0.030 # 3% bar
    ret[jump] -= rng.uniform(0.07, 0.11, size=jump.sum())
    close = 100.0 * np.cumprod(1 + ret)
    open_ = np.empty_like(close)
    open_[0], open_[1:] = close[0], close[:-1]
    hi = np.maximum(open_, close) * (1 + rng.uniform(0.0, 0.002, n_bars))
    lo = np.minimum(open_, close) * (1 - rng.uniform(0.0, 0.002, n_bars))
    return np.column_stack([open_, hi, lo, close])
