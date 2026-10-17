"""P0 test:Investigator + ."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ================= Investigator:hypothesis =================

def test_tracker_formation_normalizes():
    from causallens.agents.investigator import HypothesisTracker

    t = HypothesisTracker()
    snap = t.form(["H", "H", "H"])
    posts = [h["posterior"] for h in snap["hypotheses"].values()]
    assert abs(sum(posts) - 1.0) < 1e-3 # 4
    assert all(abs(p - 1 / 3) < 1e-3 for p in posts) # ()


def test_tracker_bayesian_direction_and_clamp():
    from causallens.agents.investigator import HypothesisTracker

    t = HypothesisTracker()
    t.form(["A", "B"])
    t.update([{"hypothesis_id": "H1", "tool_used": "x", "evidence_summary": "s", "direction": "supports", "weight": 1.0}])
    p = t.posteriors()
    assert p["H1"] > 0.5
    # evidence:weight 5
    t2 = HypothesisTracker()
    t2.form(["A", "B"])
    t2.update([{"hypothesis_id": "H1", "tool_used": "x", "evidence_summary": "s",
                "direction": "supports", "weight": 999}]) #
    assert t2.posteriors()["H1"] < 0.95 #


def test_tracker_verdict_threshold_and_inconclusive():
    from causallens.agents.investigator import HypothesisTracker

    t = HypothesisTracker()
    t.form(["A", "B"])
    v = t.verdict()
    assert v["verdict"] == "inconclusive" # evidence

    t.update([
        {"hypothesis_id": "H1", "tool_used": "x", "evidence_summary": "s", "direction": "supports", "weight": 4},
        {"hypothesis_id": "H1", "tool_used": "y", "evidence_summary": "s", "direction": "supports", "weight": 4},
        {"hypothesis_id": "H2", "tool_used": "z", "evidence_summary": "s", "direction": "against", "weight": 3},
    ])
    v = t.verdict()
    assert v["verdict"] == "confirmed" and v["winner"] == "H1"
    assert all("summary" in e for e in v["evidence"]) # evidence


# ================= investigator( LLM) =================

def test_session_investigate_finance():
    from causallens.core import ConceptPack, MicSession
    from causallens.demos import train_finance_demo
    from causallens.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "finance.json"))
    session = MicSession(detect_driver(model, feats, bg, name="inv_fin"), pack)

    result = session.investigate(n_samples=6)
    assert len(result["hypotheses"]["hypotheses"]) >= 2 # hypothesisfinding
    assert result["verdict"]["verdict"] in {"confirmed", "inconclusive", "no_hypotheses"}
    if result["verdict"]["verdict"] == "confirmed":
        assert "evidence" in result["verdict"]
    # tool
    tools = {e["tool"] for e in session.evidence_log}
    assert {"form_hypotheses", "update_hypotheses", "conclude_investigation"} <= tools


def test_session_hypothesis_tools_roundtrip():
    from causallens.core import ConceptPack, MicSession
    from causallens.demos import train_finance_demo
    from causallens.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    session = MicSession(detect_driver(model, feats, bg), ConceptPack())
    snap = session.form_hypotheses(["hypothesis:", "hypothesis:feature"])
    assert "H1" in snap["hypotheses"]
    session.update_hypotheses(
        [{"hypothesis_id": "H1", "tool_used": "get_attribution", "evidence_summary": "32%",
          "direction": "supports", "weight": 2}]
    )
    out = session.get_hypotheses()
    assert out["hypotheses"]["H1"]["n_evidence"] == 1
    v = session.conclude_investigation()
    assert v["verdict"] in {"confirmed", "inconclusive"}


# ================= :German Credit =================

def test_credit_demo_dirty_path():
    pytest.importorskip("lightgbm")
    try:
        from causallens.demos import train_credit_demo

        model, feats, bg, auc = train_credit_demo()
    except Exception as e: # (OpenML )
        if "network" in str(e).lower() or "HTTP" in str(e) or "URLError" in str(e):
            pytest.skip(f"OpenML : {e}")
        raise

    assert auc > 0.65 # credit-g LGBM
    assert any(f.startswith("was_missing_") for f in feats) # feature
    assert "checking_status" in feats and "purpose" in feats # feature

    from causallens.core import ConceptPack, MicSession
    from causallens.drivers import detect_driver

    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "credit.json"))
    session = MicSession(detect_driver(model, feats, bg, name="credit"), pack)
    report = session.offline_report(0)
    for kw in ["", "", ""]:
        assert kw in report #

    # attribution:checking_status Top5(credit-g )
    attr = session.get_attribution(0)
    top5 = [a["feature"] for a in attr["attributions"][:5]]
    assert "checking_status" in top5
