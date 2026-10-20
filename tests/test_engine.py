""" 1/2 test:mic ask reasoningengine + + concept pack."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ================= mic ask: =================

@pytest.fixture(scope="module")
def geo_session():
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_geo_demo
    from intervense.drivers import detect_driver

    model, feats, bg, _ = train_geo_demo()
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "geo.json"))
    return MicSession(detect_driver(model, feats, bg, name="geo"), pack)


def test_ask_deterministic_scale(geo_session):
    from intervense.ask import AskEngine

    ans = AskEngine(geo_session, llm=None).ask("if precipitation increases by 50%, how does landslide risk change for the typical sample?")
    for kw in ["reasoning", "do(", "answer", "boundary"]:
        assert kw in ans.lower() or kw in ans, f"missing {kw}"
    # +50% intervention
    assert "+50%" in ans


def test_ask_deterministic_quantile_and_sample(geo_session):
    from intervense.ask import AskEngine

    ans = AskEngine(geo_session, llm=None).ask("if slope rises to the 90th percentile,how does risk change for the safest sample?")
    assert "do(" in ans


def test_ask_deterministic_unparseable(geo_session):
    from intervense.ask import AskEngine

    with pytest.raises(ValueError):
        AskEngine(geo_session, llm=None).ask("what is the weather today")


def test_ask_only_mentioned_features(geo_session):
    """:resultinterventionproblemfeature."""
    from intervense.ask import AskEngine

    eng = AskEngine(geo_session, llm=None)
    plan = {"assignments": [
        {"feature": "precipitation", "op": "scale", "value": 1.5},
        {"feature": "slope", "op": "scale", "value": 1.2}, # ,
    ], "sample_selector": "median"}
    kept = eng._filter_mentioned(plan["assignments"], "ifprecipitation increases by 50%")
    assert [a["feature"] for a in kept] == ["precipitation"]


# ================= =================

def test_multiclass_iris_full_pipeline():
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_iris_demo
    from intervense.drivers import detect_driver
    from intervense.analysis import search_counterfactuals

    model, feats, bg, acc = train_iris_demo()
    assert acc > 0.9
    drv = detect_driver(model, feats, bg, name="iris")
    assert drv.n_classes == 3

    pack = ConceptPack.from_dict({"domain": "botany", "features": {
        "sepal_length": {"concept": ""}, "sepal_width": {"concept": ""},
        "petal_length": {"concept": ""}, "petal_width": {"concept": ""}}})
    session = MicSession(drv, pack)

    # :attribution
    session.set_target_class(2)
    assert drv.target_class == 2
    attr = session.get_attribution(0)
    assert len(attr["attributions"]) == 4
    top = attr["attributions"][0]["feature"]
    assert top in ("petal_width", "petal_length") #

    # counterfactual = argmax
    idx = int(np.argmax(drv.predict(bg)))
    base_label = int(drv.predict_label(bg[idx:idx + 1])[0])
    paths = search_counterfactuals(drv, bg[idx])
    assert paths
    x2 = drv.intervene(bg[idx], {f: v[1] for f, v in paths[0].changed_features.items()})
    assert int(drv.predict_label(x2.reshape(1, -1))[0]) != base_label


def test_binary_regression_unchanged(finance_env=None):
    """:."""
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_finance_demo
    from intervense.drivers import detect_driver
    from intervense.analysis import search_counterfactuals

    model, feats, bg, _ = train_finance_demo()
    drv = detect_driver(model, feats, bg, name="fin_guard")
    assert drv.n_classes == 2
    idx = int(np.argmax(drv.predict(bg)))
    paths = search_counterfactuals(drv, bg[idx])
    assert paths and paths[0].flipped


# ================= concept pack =================

def test_draft_fallback_without_llm():
    from intervense.concepts_draft import draft_concept_pack

    pack = draft_concept_pack(["f1", "f2"], llm=None, domain_hint="test")
    assert pack["features"]["f1"]["concept"] == "f1" # LLM feature


def test_attach_concepts_written(tmp_path):
    from intervense.concepts_draft import draft_concept_pack

    # LLM( JSON)verification
    class FakeLLM:
        def chat(self, messages, temperature=0.1, tools=None):
            return {"choices": [{"message": {"content":
                '{"domain":"t","name":"n","features":{"a":{"concept":"","direction":"?"}}}'}}]}

    pack = draft_concept_pack(["a", "b"], llm=FakeLLM(), domain_hint="t")
    assert pack["features"]["a"]["concept"] == ""
    assert pack["features"]["b"]["concept"] == "b" # LLM
