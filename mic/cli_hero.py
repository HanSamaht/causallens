"""`python -m mic hero` —— 30 秒英雄演示：一个成交假设虚增回测 8 倍。

开源落地页的首屏命令：无需任何数据、GPU、API Key，单机秒级跑完，
输出一个有冲击力的因果审计故事（可截图、可录 GIF）。
"""
from __future__ import annotations

import json


def run_hero() -> None:
    from mic.backtest import make_synthetic_4h
    mic_line = "=" * 64
    print(mic_line)
    print("  MIC 因果推演引擎 · 30 秒演示：审讯一个回测引擎")
    print("  核心句：相关性告诉你什么一起发生，因果性告诉你改变什么会发生什么")
    print(mic_line)

    from mic.core import ConceptPack, MicSession
    from mic.drivers.backtest import BacktestDriver

    drv = BacktestDriver(make_synthetic_4h(8000, seed=7), name="bt_engine")
    session = MicSession(drv, ConceptPack.load("concepts/backtest_zh.json"))

    print("\n[1] 表象：这个策略回测 PF≈5、胜率 56% —— 看起来是棵摇钱树\n")
    ladder = drv.run(fill_market=0.0)
    print(f"    回测器报告:  胜率 {ladder['win_rate']:.0%} | PF {ladder['pf']:.2f} | 单笔 {ladder['avg_pnl_pct']:+.2%}")

    print("\n[2] 审讯：把『成交语义』这个隐藏假设翻转（do(成交:=市价)）\n")
    r = session.intervene(2, {"fill_market": 0.0})  # 样本2=市价基线
    market = drv.run(fill_market=1.0)
    print(f"    干预结果:    胜率 {market['win_rate']:.0%} | PF {market['pf']:.2f} | 单笔 {market['avg_pnl_pct']:+.2%}")
    ratio = ladder["pf"] / max(market["pf"], 1e-3)
    print(f"\n    ⚡ 唯一变量对照 → PF 虚增 {ratio:.1f} 倍")
    print(f"    ⚡ 病灶：止损触发时价格早已跌穿止损线（平均跌穿深度 {market['avg_gap_through']:.1%}），")
    print(f"       回测器却假设能按止损线价格成交")

    print("\n[3] 压力测试：edge 对成交劣化的生存曲线\n")
    s = session.slippage_stress()
    for c in s["curve"][:4]:
        bar = "█" * int(c["pf"] * 10)
        print(f"    滑点 {c['slippage_bps']:>4.0f} bps: PF={c['pf']:.2f} {bar}")
    print(f"\n    ⚠ {s['verdict']}")

    print("\n[4] 反事实手术：哪个最小改动让它翻转？\n")
    cf = session.run_counterfactual(2)
    for i, p in enumerate(cf["paths"][:2], 1):
        changes = "；".join(f"{v['concept']} {v['from']:+.2f}→{v['to']:+.2f}" for v in p["changed_features"].values())
        print(f"    路径{i}: {changes} → PF {p['prediction_after']:.2f}")

    print(f"\n{mic_line}")
    print("  这不是解释，是干预推演：给定任何黑箱系统（模型/回测器/决策日志），")
    print("  MIC 构建因果图，告诉你——改变 X，Y 会怎样。")
    print("  试试: python -m mic report --demo finance --concepts concepts/finance_zh.json")
    print(mic_line)


def cmd_hero(args):
    run_hero()
