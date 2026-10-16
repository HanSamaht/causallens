"""评估器专项体检：滑点压力测试 / 假设缺口 / 回测-实盘忠实性。

这三个工具回答的真实问题（来自实战案例）：
  1. slippage_stress：PF 对成交劣化的敏感曲线。PF 在几十 bps 内塌缩到 1
     = "edge 依赖成交假设"的特征信号——本可以提前报警。
  2. assumption_gap：乐观假设 vs 现实语义的虚增倍数（阶梯 PF / 市价 PF）。
  3. live_faithfulness：回测数字 vs 实盘数字的吻合度——评估器的忠实性。
"""
from __future__ import annotations

import numpy as np

from ..drivers.backtest import BacktestDriver


def slippage_stress(
    driver: BacktestDriver,
    slippage_grid_bps: tuple[float, ...] = (0, 5, 10, 25, 50, 75, 100, 150, 200),
    fill_market: float = 1.0,
) -> dict:
    """PF 对滑点的生存曲线（现实市价语义下）。"""
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
            f"即使零滑点 PF={pf0:.2f} 也未过盈亏平衡线——现实成交语义下 edge 为负，"
            "全部'盈利'由成交假设制造"
        )
        fragility = "脆弱"
    else:
        fragility = "脆弱" if (break_bps is not None and break_bps <= 25) else \
                    ("一般" if break_bps is not None else "稳健")
        verdict = (
            f"edge 对成交劣化{fragility}："
            + (f"滑点达 {break_bps:.0f} bps 时 PF 跌破 1" if break_bps is not None else "全网格内 PF>1")
            + ("——这是『edge 依赖成交假设』的特征信号，回测数字不可直接采信" if fragility == "脆弱" else "")
        )
    return {
        "curve": curve,
        "pf_at_zero_slippage": pf0,
        "break_even_slippage_bps": break_bps,
        "verdict": verdict,
        "note": "参考：主流现货对中小资金单边冲击约 5-30bps；PF 若在此区间塌缩，说明盈利主要来自成交假设",
    }


def assumption_gap(driver: BacktestDriver) -> dict:
    """唯一变量对照：阶梯价记账 vs 市价成交——整个'回测-实盘落差'的手术台。"""
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
            f"乐观成交假设使 PF 虚增 {inflation:.1f} 倍"
            + ("——回测器存在系统性成交假设缺陷，所有历史回测数字的绝对水平不可信" if severe else "")
        ),
        "note": "同数据、同参数、唯一差异 do(成交语义)；跌穿深度=" + f"{ladder['avg_gap_through']:.2%}"
        + "（止损触发时价格已跌破止损线的平均深度）",
    }


def live_faithfulness(backtest: dict, live: dict) -> dict:
    """回测 vs 实盘吻合度——评估器自己的忠实性考卷。

    用法：把你在两个世界实测的指标喂进来，如
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
    verdict = "吻合" if pf_gap <= 0.15 else ("基本吻合" if pf_gap <= 0.35 else "失真")
    return {
        "metrics": gaps,
        "verdict": verdict,
        "note": (
            "参考标尺：1m重放 PF 1.09 vs 实盘 0.94（相对差 -14%，'吻合'）；"
            "理想回测 PF 10-27 vs 实盘 0.94（失真 >90%）——两者的差就是成交假设的代价"
        ),
    }
