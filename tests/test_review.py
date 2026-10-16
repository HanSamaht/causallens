"""DecisionLogDriver + 因果图人工修正 测试。"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ================= DecisionLogDriver =================

@pytest.fixture(scope="module")
def agent_session():
    from mic.core import ConceptPack, MicSession
    from mic.demos import AGENT_LOG_FEATURES, make_agent_logs
    from mic.drivers.decision_log import DecisionLogDriver

    X, y = make_agent_logs(n=2500)
    drv = DecisionLogDriver(X, y, AGENT_LOG_FEATURES, name="agent_test")
    pack = ConceptPack.from_dict({"domain": "credit", "features": {
        "credit_score": {"concept": "信用分"}, "debt_ratio": {"concept": "负债率"},
        "income_k": {"concept": "收入"}, "loan_amount_k": {"concept": "贷款额"},
        "employment_years": {"concept": "工作年限"},
        "region_code": {"concept": "地区编号", "role": "spurious"}}})
    return MicSession(drv, pack)


def test_surrogate_fidelity(agent_session):
    drv = agent_session.driver
    assert drv.kind == "decision-log"
    assert drv.fidelity["action_agreement"] > 0.85   # 行为克隆学到了政策
    assert drv.fidelity["n_logs"] == 2500
    assert "surrogate_fidelity" in drv.describe()


def test_decision_log_interrogation_finds_hidden_bias(agent_session):
    """核心场景：无 predict 函数的系统，隐藏偏见从日志中被审讯出来。"""
    audit = agent_session.audit(None)
    codes = {f["code"] for f in audit["findings"]}
    assert "spurious_dependence" in codes  # region_code 被抓到
    assert agent_session.driver.fidelity["action_agreement"] > 0.85


def test_decision_log_ask_carries_fidelity_boundary(agent_session):
    from mic.ask import AskEngine

    ans = AskEngine(agent_session, llm=None).ask("如果负债率增加50%，最典型的样本批准概率怎么变")
    assert "动作一致率" in ans          # 保真度标签自动附在边界里
    assert "不是对现实世界的预测" in ans
    assert "负债率" in ans


def test_decision_log_counterfactual(agent_session):
    from mic.analysis import search_counterfactuals

    drv = agent_session.driver
    idx = int(np.argmax(drv.predict(drv.background)))
    paths = search_counterfactuals(drv, drv.background[idx])
    assert paths  # 替代模型反事实照常工作


# ================= 因果图人工修正 =================

@pytest.fixture(scope="module")
def fin_session():
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_finance_demo
    from mic.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json"))
    return MicSession(detect_driver(model, feats, bg, name="fin_review"), pack)


def test_graph_export_edit_apply(tmp_path, fin_session):
    from mic.core.graph_review import apply_corrections, export_graph

    out = tmp_path / "g.json"
    info = export_graph(fin_session, str(out))
    assert info["n_edges"] > 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "_说明" in doc  # 内嵌编辑说明
    # 专家编辑：确认 + 否决 + 新增（1 条真效应 + 1 条无效应 + 1 条非法）
    for e in doc["edges"]:
        if e["source"] == "momentum" and e["target"] == "prediction":
            e.update(status="confirmed", note="领域共识")
        if e["source"] == "month_index" and e["target"] == "prediction":
            e.update(status="rejected", note="不应存在")
    doc["edges"] += [
        {"source": "liquidity", "target": "prediction", "status": "added", "note": "先验"},
        {"source": "size", "target": "momentum", "status": "added", "note": "先验"},
        {"source": "ghost_feature", "target": "prediction", "status": "added", "note": "手滑"},
    ]
    edited = tmp_path / "g_edited.json"
    edited.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    result = apply_corrections(fin_session, str(edited))
    assert len(result["confirmed"]) == 1
    assert len(result["rejected"]) >= 1
    assert len(result["invalid"]) == 1
    verdicts = {a["edge"]: a["verification"]["verdict"] for a in result["added"]}
    assert verdicts["liquidity->prediction"].startswith("实验支持")     # 真效应先验通过
    assert verdicts["size->momentum"].startswith("实验不支持")          # 假先验被实验否决
    # 修正后的图挂在 session 上，含验证状态
    kinds = {e["kind"] for e in fin_session.corrected_graph["edges"]}
    assert "user_prior_verified" in kinds
    assert "user_prior_rejected" in kinds
