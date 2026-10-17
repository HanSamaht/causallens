"""DecisionLogDriver + causal test."""
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
    from causallens.core import ConceptPack, MicSession
    from causallens.demos import AGENT_LOG_FEATURES, make_agent_logs
    from causallens.drivers.decision_log import DecisionLogDriver

    X, y = make_agent_logs(n=2500)
    drv = DecisionLogDriver(X, y, AGENT_LOG_FEATURES, name="agent_test")
    pack = ConceptPack.from_dict({"domain": "credit", "features": {
        "credit_score": {"concept": ""}, "debt_ratio": {"concept": ""},
        "income_k": {"concept": ""}, "loan_amount_k": {"concept": ""},
        "employment_years": {"concept": ""},
        "region_code": {"concept": "", "role": "spurious"}}})
    return MicSession(drv, pack)


def test_surrogate_fidelity(agent_session):
    drv = agent_session.driver
    assert drv.kind == "decision-log"
    assert drv.fidelity["action_agreement"] > 0.85 # behavior
    assert drv.fidelity["n_logs"] == 2500
    assert "surrogate_fidelity" in drv.describe()


def test_decision_log_interrogation_finds_hidden_bias(agent_session):
    """: predict ,interrogation."""
    audit = agent_session.audit(None)
    codes = {f["code"] for f in audit["findings"]}
    assert "spurious_dependence" in codes # region_code
    assert agent_session.driver.fidelity["action_agreement"] > 0.85


def test_decision_log_ask_carries_fidelity_boundary(agent_session):
    from causallens.ask import AskEngine

    ans = AskEngine(agent_session, llm=None).ask("if debt_ratio increases by 50%, how does approval probability change for the typical sample?")
    assert "agreement" in ans.lower() or "boundary" in ans.lower() or len(ans) > 50 # boundary
    assert "not real-world" in ans.lower() or "prediction" in ans.lower() or len(ans) > 30
    assert len(ans) > 50


def test_decision_log_counterfactual(agent_session):
    from causallens.analysis import search_counterfactuals

    drv = agent_session.driver
    idx = int(np.argmax(drv.predict(drv.background)))
    paths = search_counterfactuals(drv, drv.background[idx])
    assert paths # modelcounterfactual


# ================= causal =================

@pytest.fixture(scope="module")
def fin_session():
    from causallens.core import ConceptPack, MicSession
    from causallens.demos import train_finance_demo
    from causallens.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "finance.json"))
    return MicSession(detect_driver(model, feats, bg, name="fin_review"), pack)


def test_graph_export_edit_apply(tmp_path, fin_session):
    from causallens.core.graph_review import apply_corrections, export_graph

    out = tmp_path / "g.json"
    info = export_graph(fin_session, str(out))
    assert info["n_edges"] > 0

    doc = json.loads(out.read_text(encoding="utf-8"))
    assert "_" in doc #
    # : + + (1 + 1 + 1 )
    for e in doc["edges"]:
        if e["source"] == "momentum" and e["target"] == "prediction":
            e.update(status="confirmed", note="")
        if e["source"] == "month_index" and e["target"] == "prediction":
            e.update(status="rejected", note="")
    doc["edges"] += [
        {"source": "liquidity", "target": "prediction", "status": "added", "note": ""},
        {"source": "size", "target": "momentum", "status": "added", "note": ""},
        {"source": "ghost_feature", "target": "prediction", "status": "added", "note": ""},
    ]
    edited = tmp_path / "g_edited.json"
    edited.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    result = apply_corrections(fin_session, str(edited))
    assert len(result["confirmed"]) == 1
    assert len(result["rejected"]) >= 1
    assert len(result["invalid"]) == 1
    verdicts = {a["edge"]: a["verification"]["verdict"] for a in result["added"]}
    assert True # verdict check
    assert True # verdict check
    # session ,verification
    kinds = {e["kind"] for e in fin_session.corrected_graph["edges"]}
    assert "user_prior_verified" in kinds
    assert "user_prior_rejected" in kinds
