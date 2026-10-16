"""P0 两项测试：Investigator 循环 + 脏数据基准。"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ================= Investigator：假设追踪器数学 =================

def test_tracker_formation_normalizes():
    from mic.agents.investigator import HypothesisTracker

    t = HypothesisTracker()
    snap = t.form(["H甲陈述", "H乙陈述", "H丙陈述"])
    posts = [h["posterior"] for h in snap["hypotheses"].values()]
    assert abs(sum(posts) - 1.0) < 1e-3  # 快照按4位小数舍入
    assert all(abs(p - 1 / 3) < 1e-3 for p in posts)  # 均匀先验(舍入容差)


def test_tracker_bayesian_direction_and_clamp():
    from mic.agents.investigator import HypothesisTracker

    t = HypothesisTracker()
    t.form(["A", "B"])
    t.update([{"hypothesis_id": "H1", "tool_used": "x", "evidence_summary": "s", "direction": "supports", "weight": 1.0}])
    p = t.posteriors()
    assert p["H1"] > 0.5
    # 单条证据不能掀翻全局：weight 钳制在 5
    t2 = HypothesisTracker()
    t2.form(["A", "B"])
    t2.update([{"hypothesis_id": "H1", "tool_used": "x", "evidence_summary": "s",
                "direction": "supports", "weight": 999}])  # 恶意权重
    assert t2.posteriors()["H1"] < 0.95  # 钳制生效


def test_tracker_verdict_threshold_and_inconclusive():
    from mic.agents.investigator import HypothesisTracker

    t = HypothesisTracker()
    t.form(["A", "B"])
    v = t.verdict()
    assert v["verdict"] == "inconclusive"  # 无证据不许下结论

    t.update([
        {"hypothesis_id": "H1", "tool_used": "x", "evidence_summary": "s", "direction": "supports", "weight": 4},
        {"hypothesis_id": "H1", "tool_used": "y", "evidence_summary": "s", "direction": "supports", "weight": 4},
        {"hypothesis_id": "H2", "tool_used": "z", "evidence_summary": "s", "direction": "against", "weight": 3},
    ])
    v = t.verdict()
    assert v["verdict"] == "confirmed" and v["winner"] == "H1"
    assert all("summary" in e for e in v["evidence"])  # 证据链留痕


# ================= 确定性调查员（无 LLM） =================

def test_session_investigate_finance():
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_finance_demo
    from mic.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json"))
    session = MicSession(detect_driver(model, feats, bg, name="inv_fin"), pack)

    result = session.investigate(n_samples=6)
    assert len(result["hypotheses"]["hypotheses"]) >= 2  # 竞争假设而非单发现
    assert result["verdict"]["verdict"] in {"confirmed", "inconclusive", "no_hypotheses"}
    if result["verdict"]["verdict"] == "confirmed":
        assert "evidence" in result["verdict"]
    # 工具留痕
    tools = {e["tool"] for e in session.evidence_log}
    assert {"form_hypotheses", "update_hypotheses", "conclude_investigation"} <= tools


def test_session_hypothesis_tools_roundtrip():
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_finance_demo
    from mic.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    session = MicSession(detect_driver(model, feats, bg), ConceptPack())
    snap = session.form_hypotheses(["假设一：动量主导", "假设二：伪特征主导"])
    assert "H1" in snap["hypotheses"]
    session.update_hypotheses(
        [{"hypothesis_id": "H1", "tool_used": "get_attribution", "evidence_summary": "动量占比32%",
          "direction": "supports", "weight": 2}]
    )
    out = session.get_hypotheses()
    assert out["hypotheses"]["H1"]["n_evidence"] == 1
    v = session.conclude_investigation()
    assert v["verdict"] in {"confirmed", "inconclusive"}


# ================= 脏数据基准：German Credit =================

def test_credit_demo_dirty_path():
    pytest.importorskip("lightgbm")
    try:
        from mic.demos import train_credit_demo

        model, feats, bg, auc = train_credit_demo()
    except Exception as e:  # 网络不可达时跳过（OpenML 下载）
        if "network" in str(e).lower() or "HTTP" in str(e) or "URLError" in str(e):
            pytest.skip(f"OpenML 不可达: {e}")
        raise

    assert auc > 0.65  # credit-g 的 LGBM 合理区间
    assert any(f.startswith("was_missing_") for f in feats)  # 缺失指示特征在场
    assert "checking_status" in feats and "purpose" in feats  # 真实类别特征原样进管线

    from mic.core import ConceptPack, MicSession
    from mic.drivers import detect_driver

    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "credit_zh.json"))
    session = MicSession(detect_driver(model, feats, bg, name="credit"), pack)
    report = session.offline_report(0)
    for kw in ["支票账户状态", "贷款金额", "信用历史"]:
        assert kw in report  # 领域概念接地成功

    # 归因合理性：checking_status 应进 Top5（credit-g 文献共识的最强因子）
    attr = session.get_attribution(0)
    top5 = [a["feature"] for a in attr["attributions"][:5]]
    assert "checking_status" in top5
