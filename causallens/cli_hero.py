"""`python -m mic hero` —— 30 :hypothesisbacktest 8 .

:,GPU,API Key,,
outputhascausalaudit(, GIF).
"""
from __future__ import annotations

import json


def run_hero() -> None:
    """Run hero."""
    from causallens.backtest import make_synthetic_4h
    mic_line = "=" * 64
    print(mic_line)
    print(" MIC causal reasoningengine · 30 :interrogationbacktestengine")
    print(" :Correlation tells you what happens together; causality tells you what happens when you intervene")
    print(mic_line)

    from causallens.core import ConceptPack, MicSession
    from causallens.drivers.backtest import BacktestDriver

    drv = BacktestDriver(make_synthetic_4h(8000, seed=7), name="bt_engine")
    session = MicSession(drv, ConceptPack.load("concepts/backtest.json"))

    print("\n[1] :thisbacktest PF≈5, 56% —— is\n")
    ladder = drv.run(fill_market=0.0)
    print(f" backtestreport: {ladder['win_rate']:.0%} | PF {ladder['pf']:.2f} | {ladder['avg_pnl_pct']:+.2%}")

    print("\n[2] interrogation:'fill semantics'thishypothesis(do(:=))\n")
    r = session.intervene(2, {"fill_market": 0.0}) # sample2=
    market = drv.run(fill_market=1.0)
    print(f" interventionresult: {market['win_rate']:.0%} | PF {market['pf']:.2f} | {market['avg_pnl_pct']:+.2%}")
    ratio = ladder["pf"] / max(market["pf"], 1e-3)
    print(f"\n ⚡ -> PF {ratio:.1f} ")
    print(f" ⚡ :(depth {market['avg_gap_through']:.1%}),")
    print(f" backtesthypothesis")

    print("\n[3] test:edge \n")
    s = session.slippage_stress()
    for c in s["curve"][:4]:
        bar = "█" * int(c["pf"] * 10)
        print(f" {c['slippage_bps']:>4.0f} bps: PF={c['pf']:.2f} {bar}")
    print(f"\n ⚠ {s['verdict']}")

    print("\n[4] counterfactual:most?\n")
    cf = session.run_counterfactual(2)
    for i, p in enumerate(cf["paths"][:2], 1):
        changes = ";".join(f"{v['concept']} {v['from']:+.2f}->{v['to']:+.2f}" for v in p["changed_features"].values())
        print(f" {i}: {changes} -> PF {p['prediction_after']:.2f}")

    print(f"\n{mic_line}")
    print(" thisnot,isinterventionreasoning:(model/backtest/log),")
    print(" MIC causal graph,—— X,Y .")
    print(" : python -m mic report --demo finance --concepts concepts/finance.json")
    print(mic_line)


def cmd_hero(args):
    """Cmd hero."""
    run_hero()
