"""evidence ledger + case archivetest: / chain-hash tamper-proof / / ."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def ledger(tmp_path):
    from intervense.ledger import Ledger

    return Ledger(str(tmp_path / "test_ledger.db"))


def test_case_open_idempotent(ledger):
    a = ledger.open_case("m", "v1")
    b = ledger.open_case("m", "v1")
    assert a == b
    assert ledger.open_case("m", "v2") != a


def test_evidence_chain_and_tamper_detection(ledger):
    case = ledger.open_case("m", "v1")
    for i in range(5):
        ledger.append_evidence(case, f"tool_{i}", {"i": i})
    v = ledger.verify_chain(case)
    assert v["intact"] and v["n_evidence"] == 5
    # 3 ->
    ledger.conn.execute("UPDATE evidence SET payload_json='{\"i\":999}' WHERE id=3")
    ledger.conn.commit()
    v2 = ledger.verify_chain(case)
    assert not v2["intact"] and v2["broken_at_id"] == 3


def test_hypothesis_bi_temporal_supersede(ledger):
    c1 = ledger.open_case("m", "v1")
    c2 = ledger.open_case("m", "v2")
    ledger.record_hypothesis(c1, "modelsuspiciousfeature", "confirmed", 0.86)
    priors_before = ledger.hypothesis_priors("m")
    assert priors_before[0]["still_valid"] is True
    # v2 hypothesis -> v1
    ledger.record_hypothesis(c2, "modelsuspiciousfeature", "rejected", 0.12)
    priors = ledger.hypothesis_priors("m")
    v2row = [p for p in priors if p["model_version"] == "v2"][0]
    v1row = [p for p in priors if p["model_version"] == "v1"][0]
    assert v2row["still_valid"] and v1row["valid_to"] is not None


def test_session_autopersist_and_priors(tmp_path):
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_finance_demo
    from intervense.drivers import detect_driver
    from intervense.ledger import Ledger

    led = Ledger(str(tmp_path / "l.db"))
    model, feats, bg, _ = train_finance_demo()
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "finance.json"))

    s1 = MicSession(detect_driver(model, feats, bg, name="led_test"), pack)
    s1.attach_ledger(led, "v1")
    s1.get_attribution(0)
    s1.investigate(n_samples=5)
    assert len(s1.evidence_log) > 3
    summary = led.case_summary(s1.case_id)
    assert summary["chain"]["intact"]
    assert len(summary["hypotheses"]) >= 2

    # v2 v1 hypothesis
    s2 = MicSession(detect_driver(model, feats, bg, name="led_test"), pack)
    s2.attach_ledger(led, "v2")
    brief = s2.case_briefing()
    assert len(brief["priors"]) >= 2
    r = s2.investigate(n_samples=5)
    assert "case_priors" in r


def test_correction_recorded(tmp_path):
    from intervense.core import ConceptPack, MicSession
    from intervense.core.graph_review import apply_corrections, export_graph
    from intervense.demos import train_finance_demo
    from intervense.drivers import detect_driver
    from intervense.ledger import Ledger

    import json

    led = Ledger(str(tmp_path / "l2.db"))
    model, feats, bg, _ = train_finance_demo()
    s = MicSession(detect_driver(model, feats, bg, name="corr_test"),
                   ConceptPack.load(os.path.join(_HERE, "concepts", "finance.json")))
    s.attach_ledger(led, "v1")
    out = tmp_path / "g.json"
    export_graph(s, str(out))
    doc = json.loads(out.read_text(encoding="utf-8"))
    doc["edges"][0]["status"] = "confirmed"
    doc["edges"] += [{"source": "liquidity", "target": "prediction", "status": "added", "note": ""}]
    out.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    apply_corrections(s, str(out))
    summary = led.case_summary(s.case_id)
    ops = {c["operation"] for c in summary["corrections"]}
    assert {"confirmed", "added"} <= ops
