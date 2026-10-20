"""Backtest stress audits: fragility, assumption gap, live fidelity.

Three tools that answer three production questions:
  1. slippage_stress: at what cost in bps does PF drop to 1, which
     measures how much of the edge is really an execution assumption
  2. assumption_gap: ladder accounting versus market fill, the PF
     inflation ratio between the two
  3. live_faithfulness: backtest promises versus live results
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
            f"PF={pf0:.2f} even before costs, there is no edge under any "
            "fill hypothesis"
        )
        fragility = ""
    else:
        fragility = "" if (break_bps is not None and break_bps <= 25) else \
                    ("" if break_bps is not None else "")
        verdict = (
            "edge verdict:"
            + (f" breaks at {break_bps:.0f} bps" if break_bps is not None else " PF stays above 1")
            + (", the edge is mostly an execution assumption, not a market signal" if fragility == "" else "")
        )
    return {
        "curve": curve,
        "pf_at_zero_slippage": pf0,
        "break_even_slippage_bps": break_bps,
        "verdict": verdict,
        "note": ":in 5-30bps;PF inthis,hypothesis",
    }


def assumption_gap(driver: BacktestDriver) -> dict:
    """Assumption gap: ladder price accounting versus market fill, the heart of any backtest."""
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
            f"fill hypothesis inflates PF {inflation:.1f} times "
            + (", every historical backtest on this hypothesis is unusable" if severe else "")
        ),
        "note": ("the only parameter that differs is fill semantics, an intervention; "
                 f"gap through depth {ladder['avg_gap_through']:.2%}"),
    }


def live_faithfulness(backtest: dict, live: dict) -> dict:
    """Backtest versus live: measure whether the backtest kept its promise.

    Example call:
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
            "at 1m granularity PF 1.09 versus live 0.94, a 14 percent miss, borderline;"
            " backtest PF 10 to 27 versus live 0.94 is over 90 percent inflation, the backtest hypothesis itself is broken"
        ),
    }
