"""`python -m intervense hero`: a self-contained end-to-end demo.

Runs a backtest engine interrogation in one pass: baseline replay, a
fill semantics intervention, a slippage stress test, and a
counterfactual repair path. Works offline on CPU with no API key. Every
printed line carries a causal verdict rather than a plain number.
"""
from __future__ import annotations


def run_hero() -> None:
    """Run the hero demo."""
    from intervense.backtest import make_synthetic_4h
    from intervense.core import ConceptPack, MicSession
    from intervense.drivers.backtest import BacktestDriver

    line = "=" * 64
    print(line)
    print(" Intervense causal reasoning engine, backtest engine interrogation")
    print(" Correlation tells you what happens together;")
    print(" causality tells you what happens when you intervene")
    print(line)

    drv = BacktestDriver(make_synthetic_4h(8000, seed=7), name="bt_engine")
    session = MicSession(drv, ConceptPack.load("concepts/backtest.json"))

    print("\n[1] Baseline: this backtest reports PF around 5 with a 56% win rate. Is it real?\n")
    ladder = drv.run(fill_market=0.0)
    print(f" backtest report: win rate {ladder['win_rate']:.0%} | PF {ladder['pf']:.2f} "
          f"| avg pnl {ladder['avg_pnl_pct']:+.2%}")

    print("\n[2] Intervention: test the fill semantics hypothesis, do(fill_market := 1.0)\n")
    session.intervene(2, {"fill_market": 0.0})  # pin sample 2 as the anchor
    market = drv.run(fill_market=1.0)
    print(f" intervention result: win rate {market['win_rate']:.0%} | PF {market['pf']:.2f} "
          f"| avg pnl {market['avg_pnl_pct']:+.2%}")
    ratio = ladder["pf"] / max(market["pf"], 1e-3)
    print(f"\n profit factor inflated {ratio:.1f} times by the fill assumption")
    print(f" mechanism: stop orders fill through gaps with depth "
          f"{market['avg_gap_through']:.1%} of the time")
    print(" the backtest hypothesis does not survive the intervention")

    print("\n[3] Stress test: how far does the edge survive slippage?\n")
    s = session.slippage_stress()
    for c in s["curve"][:4]:
        bar = "#" * int(c["pf"] * 10)
        print(f" {c['slippage_bps']:>4.0f} bps: PF={c['pf']:.2f} {bar}")
    print(f"\n verdict: {s['verdict']}")

    print("\n[4] Counterfactual: what changes would repair this backtest?\n")
    cf = session.run_counterfactual(2)
    for i, p in enumerate(cf["paths"][:2], 1):
        changes = ";".join(f"{v['concept']} {v['from']:+.2f}->{v['to']:+.2f}"
                           for v in p["changed_features"].values())
        print(f" {i}: {changes} -> PF {p['prediction_after']:.2f}")

    print(f"\n{line}")
    print(" This was not attribution, it was intervention reasoning over an engine")
    print(" with inputs, outputs, and explicit assumptions. The same loop applies")
    print(" to any model, backtest engine, or decision log.")
    print(" Next: python -m intervense report --demo finance --concepts concepts/finance.json")
    print(line)


def cmd_hero(args):
    """CLI entry for the hero subcommand."""
    run_hero()
