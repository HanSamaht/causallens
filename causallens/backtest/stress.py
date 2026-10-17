"""evaluate:test / hypothesis / backtest-.

thistoolproblem():
  1. slippage_stress:PF .PF in bps to 1
     = "edge hypothesis"feature——can.
  2. assumption_gap:hypothesis vs ( PF / PF).
  3. live_faithfulness:backtest vs ——evaluate.
"""
from __future__ import annotations

import numpy as np

from ..drivers.backtest import BacktestDriver


def slippage_stress(
    driver: BacktestDriver,
    slippage_grid_bps: tuple[float, ...] = (0, 5, 10, 25, 50, 75, 100, 150, 200),
    fill_market: float = 1.0,
) -> dict:
    """PF (below)."""
    curve = []
    for bps in slippage_grid_bps:
        m = driver.run(fill_market=fill_market, slippage_bps=float(bps))
        curve.append(
            {
                "slippage_bps": float(bps),
                "pf": round(m["pf"], 3),
                "win_rate": round(m["win_rate"], 3),
                "avg_pnl_pct": round(m["avg_pnl_pct"], 5),
                "profitable": m["pf"] > 1.0,
            }
        )
    pf0 = curve[0]["pf"]
    break_bps = next((c["slippage_bps"] for c in curve if not c["profitable"]), None)
    if pf0 <= 1.0:
        verdict = (
            f" PF={pf0:.2f} also——fill semanticsbelow edge as,"
            "all''hypothesis"
        )
        fragility = ""
    else:
        fragility = "" if (break_bps is not None and break_bps <= 25) else \
                    ("" if break_bps is not None else "")
        verdict = (
            f"edge {fragility}:"
            + (f" {break_bps:.0f} bps PF 1" if break_bps is not None else " PF>1")
            + ("——thisis'edge hypothesis'feature,backtestnotdirect" if fragility == "" else "")
        )
    return {
        "curve": curve,
        "pf_at_zero_slippage": pf0,
        "break_even_slippage_bps": break_bps,
        "verdict": verdict,
        "note": ":in 5-30bps;PF inthis,hypothesis",
    }


def assumption_gap(driver: BacktestDriver) -> dict:
    """:ladder price accounting vs market fill——'backtest-'."""
    ladder = driver.run(fill_market=0.0)
    market = driver.run(fill_market=1.0)
    inflation = ladder["pf"] / max(market["pf"], 1e-3)
    severe = inflation >= 2.0
    return {
        "ladder_fill": {k: round(v, 4) if isinstance(v, float) else v for k, v in ladder.items() if k != "trades"},
        "market_fill": {k: round(v, 4) if isinstance(v, float) else v for k, v in market.items() if k != "trades"},
        "pf_inflation_ratio": round(float(inflation), 2),
        "win_rate_gap": round(float(ladder["win_rate"] - market["win_rate"]), 4),
        "avg_gap_through": round(ladder["avg_gap_through"], 4),
        "verdict": (
            f"hypothesis PF {inflation:.1f} "
            + ("——backtestinhypothesis,allhistorybacktestnot" if severe else "")
        ),
        "note": ",parameter,difference do(fill semantics);depth=" + f"{ladder['avg_gap_through']:.2%}"
        + "(depth)",
    }


def live_faithfulness(backtest: dict, live: dict) -> dict:
    """backtest vs ——evaluate.

    :in,
    live_faithfulness({"pf": 1.09, "win_rate": 0.48, "avg_pnl_pct": 0.0076},
                      {"pf": 0.94, "win_rate": 0.45})
    """
    gaps = {}
    for k in ("pf", "win_rate", "avg_pnl_pct"):
        if k in backtest and k in live and backtest[k] not in (None, 0):
            gaps[k] = {
                "backtest": backtest[k],
                "live": live[k],
                "relative_gap": round(float(live[k] / backtest[k] - 1), 4),
            }
    pf_gap = abs(gaps.get("pf", {}).get("relative_gap", 0))
    verdict = "" if pf_gap <= 0.15 else ("" if pf_gap <= 0.35 else "")
    return {
        "metrics": gaps,
        "verdict": verdict,
        "note": (
            ":1m PF 1.09 vs 0.94( -14%,'');"
            "backtest PF 10-27 vs 0.94( >90%)——ishypothesis"
        ),
    }
