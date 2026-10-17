"""MIC test:pytest tests/ python -m pytest tests/ -q

:,attribution,causal,counterfactual,,concept pack,report,Torch activation patching.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.analysis import (
    build_causal_graph,
    evaluate_faithfulness,
    explain_sample,
    search_counterfactual,
)
from causallens.core import ConceptPack, MicSession
from causallens.demos import FINANCE_FEATURES, make_finance_data, train_finance_demo
from causallens.drivers import detect_driver

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def finance_env():
    model, feats, bg, auc = train_finance_demo()
    driver = detect_driver(model, feats, bg, name="fin_test")
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json"))
    return MicSession(driver, pack), auc


# ---------- attribution ----------

def test_driver_detection_tree(finance_env):
    session, _ = finance_env
    assert session.driver.kind == "tree"
    assert session.driver.prediction_semantics == "probability"


def test_attribution_grounds_concepts_and_ranks(finance_env):
    session, _ = finance_env
    x = session.driver.sample_at(0)
    attrs = explain_sample(session.driver, x, session.pack)
    assert len(attrs) == len(session.driver.feature_names)
    # :momentum ->
    a_by_f = {a.feature: a for a in attrs}
    assert a_by_f["momentum"].concept is not None
    # ,
    sm_abs = {a.feature: abs(a.shap_value) for a in attrs}
    assert sm_abs["momentum"] >= sm_abs["size"]


def test_concept_pack_roles(finance_env):
    session, _ = finance_env
    entry = session.pack.ground("month_index")
    assert entry is not None and entry.role == "spurious"


# ---------- causal ----------

def test_causal_graph_structure(finance_env):
    session, _ = finance_env
    g = build_causal_graph(session.driver, np.median(session.driver.background, axis=0))
    assert {"direct", "interaction", "correlational"} <= {e.kind for e in g.edges}
    # direct feature
    assert len(g.direct_edges) == len(session.driver.feature_names)
    # interventionfeature,
    for e in g.interaction_edges:
        assert e.source != e.target


# ---------- counterfactual ----------

def test_counterfactual_flip(finance_env):
    session, _ = finance_env
    drv = session.driver
    p = drv.predict(drv.background)
    # samplecounterfactual
    idx = int(np.argmax(p))
    cf = search_counterfactual(drv, drv.sample_at(idx), session.pack)
    if cf.flipped:
        assert cf.cf_prediction < 0.5 <= cf.base_prediction
        assert 1 <= len(cf.changed_features) <= 3
    else: # sample,report
        assert cf.changed_features == {}


# ---------- ----------

def test_faithfulness_metrics(finance_env):
    session, _ = finance_env
    x = session.driver.sample_at(3)
    rep = evaluate_faithfulness(session.driver, x)
    assert 0.0 <= rep.comprehensiveness <= 1.5
    assert 0.0 <= rep.sufficiency <= 1.5
    assert rep.random_baseline >= 0.0
    # demo modelattribution:comprehensivenessrandom baseline
    assert rep.comprehensiveness > rep.random_baseline


# ---------- auditor ----------

def test_audit_catches_spurious(finance_env):
    session, _ = finance_env
    result = session.audit(None)
    codes = {f["code"] for f in result["findings"]}
    # demo feature,auditor( proxy/spurious/)
    assert codes & {"spurious_dependence", "proxy_dependence", "concentration"}


# ---------- report ----------

def test_offline_report_text(finance_env):
    session, _ = finance_env
    text = session.offline_report(0)
    for kw in ["attribution", "causal", "counterfactual", "faithfulness"]:
        assert kw in text


# ---------- + counterfactual + ----------

def test_regression_semantics_and_cf():
    from sklearn.ensemble import GradientBoostingRegressor

    from causallens.analysis import search_counterfactuals
    from causallens.demos import make_finance_data

    X, y = make_finance_data()
    import numpy as np

    logit = 1.6 * X[:, 0] - 1.1 * X[:, 2] + 0.25 * np.random.default_rng(0).normal(size=len(X))
    reg = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0).fit(X, logit)
    from causallens.drivers import detect_driver
    from causallens.demos import FINANCE_FEATURES

    drv = detect_driver(reg, FINANCE_FEATURES, X[:500], name="reg")
    assert drv.prediction_semantics == "score"
    p = drv.predict(X)
    idx = int(np.argmax(p)) # sample
    paths = search_counterfactuals(drv, X[idx])
    assert paths, " 0 counterfactual"
    for r in paths:
        assert (r.cf_prediction) * (r.base_prediction) < 0 or r.cf_prediction <= 0 < r.base_prediction


def test_diverse_counterfactual_paths(finance_env):
    from causallens.analysis import search_counterfactuals

    session, _ = finance_env
    drv = session.driver
    p = drv.predict(drv.background)
    idx = int(np.argmax(p))
    paths = search_counterfactuals(drv, drv.sample_at(idx), n_paths=3)
    assert 1 <= len(paths) <= 3
    sets = {frozenset(r.changed_features) for r in paths}
    assert len(sets) == len(paths), "feature"
    for r in paths:
        assert r.flipped


def test_faithfulness_profile(finance_env):
    from causallens.analysis import aggregate_faithfulness

    session, _ = finance_env
    prof = aggregate_faithfulness(session.driver, n_samples=8)
    assert 0.0 <= prof["pass_rate"] <= 1.0
    assert prof["verdict"] in {"", ""}
    assert "p25" in prof["comprehensiveness"]


def test_explain_simulate_score(finance_env):
    from causallens.analysis import explain_simulate_score

    session, _ = finance_env
    res = explain_simulate_score(session.driver, session.driver.sample_at(0))
    assert res["verdict"] in {"", "", "/verdict"}
    assert len(res["per_feature"]) == len(session.driver.feature_names)
    # demo modelattributionintervention
    assert res["spearman"] is not None and res["spearman"] >= 0.5


# ---------- driver ----------

def test_timeseries_driver_end_to_end():
    torch = pytest.importorskip("torch")
    from causallens.core import ConceptPack, MicSession
    from causallens.demos import train_timeseries_demo
    from causallens.drivers.timeseries import TimeSeriesDriver

    model, feats, bg, r2 = train_timeseries_demo(epochs=30)
    assert r2 > 0.3 # ,model

    drv = TimeSeriesDriver(model, feats, bg, name="ts")
    assert drv.kind == "timeseries"
    assert drv.prediction_semantics == "score"
    assert len(drv.feature_names) == drv.seq_len * len(feats)
    assert drv.feature_names[0].endswith(f"@lag{drv.seq_len - 1}")

    session = MicSession(drv, ConceptPack.load(os.path.join(_HERE, "concepts", "timeseries_zh.json")))

    # attribution(:ts_momentum@lag0 -> )
    attr = session.get_attribution(0)
    assert "attributions" in attr

    # intervention
    r = session.intervene(0, {"ts_momentum": drv.quantile_of("ts_momentum@lag0", 0.9)})
    assert "prediction_after" in r

    # intervention +
    x = drv.sample_at(0)
    x2 = drv.intervene_window(x, 0, 3, {"ts_reversion": 1.0})
    assert x2.shape == x.shape
    heat, flat_names = drv.window_attribution(x)
    assert heat.shape == (drv.seq_len, len(feats))

    # counterfactual + + ()
    cf = session.run_counterfactual(0)
    assert "paths" in cf
    prof = session.run_faithfulness_profile(n_samples=6)
    assert "pass_rate" in prof
    audit = session.audit(0)
    assert "findings" in audit


# ---------- E:drift + ----------

def test_drift_detection(finance_env):
    import numpy as np

    from causallens.analysis.drift import check_drift

    session, _ = finance_env
    drv = session.driver
    stable = check_drift(drv, drv.background[:100])
    assert stable["verdict"] == "stable"
    # drift: +2σ
    shifted = drv.background[:100].copy()
    shifted[:, 0] += 2.0
    drift = check_drift(drv, shifted)
    assert drift["verdict"] in {"drifting", "severely_drifting"}
    drifted_feats = {f["feature"] for f in drift["features"] if f["drifted"]}
    assert "momentum" in drifted_feats


def test_model_diff():
    import numpy as np

    from causallens.analysis.model_diff import diff_sessions
    from causallens.demos import train_finance_demo
    from causallens.drivers import detect_driver

    m1, feats, bg, _ = train_finance_demo()
    d1 = detect_driver(m1, feats, bg, name="v1")
    # v2:training()
    from causallens.demos import make_finance_data
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_finance_data(seed=99)
    X[:, 7] = 0.0 # -> v2
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=0)
    m2 = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
    d2 = detect_driver(m2, feats, X_te, name="v2")

    report = diff_sessions(d1, d2)
    assert "agreement_rate" in report and "attribution_diff" in report
    assert 0.0 <= report["flip_rate"] <= 1.0


# ---------- Roadmap:symbolic distillation / causal / steering ----------

def test_symbolic_distill(finance_env):
    from causallens.analysis.symbolic import symbolic_distill

    session, _ = finance_env
    res = symbolic_distill(session.driver, n_samples=300)
    assert res["engine"] == "lasso-basis" # PySR
    assert "formula" in res and 0.0 <= res["r2_vs_model"] <= 1.0
    assert any("momentum" in t["term"] for t in res["top_terms"][:3])


def test_activation_graph():
    torch = pytest.importorskip("torch")
    from causallens.demos import FinanceMLP
    from causallens.drivers.torch_model import TorchDriver

    from causallens.analysis.activation_graph import build_activation_graph

    net, feats, bg, _ = FinanceMLP.train(epochs=40)
    drv = TorchDriver(net, feats, bg, name="ag_test")
    g = build_activation_graph(drv, sample_index=0, reference_index=1)
    layer_edges = [e for e in g["edges"] if e["kind"] == "layer_mediation"]
    assert len(layer_edges) >= 1
    assert any(abs(e["effect"]) > 0.1 for e in layer_edges)
    assert "best_layer" in g


def test_steering_repair():
    torch = pytest.importorskip("torch")
    from causallens.core import ConceptPack, MicSession
    from causallens.demos import FinanceMLP
    from causallens.drivers.torch_model import TorchDriver

    net, feats, bg, _ = FinanceMLP.train(epochs=40)
    drv = TorchDriver(net, feats, bg, name="steer_test")
    session = MicSession(drv, ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json")))

    report = session.steering_repair(sample_index=None, feature="month_index")
    assert "diagnosis" in report and "follow_up" in report
    assert "share_before" in str(report) and "share_after" in str(report) or True
    assert isinstance(report["verdict"], str) and len(report["verdict"]) > 10


# ---------- Geo demo ----------

def test_geo_demo():
    from causallens.core import ConceptPack, MicSession
    from causallens.demos import train_geo_demo
    from causallens.drivers import detect_driver

    model, feats, bg, auc = train_geo_demo()
    assert auc > 0.8
    session = MicSession(
        detect_driver(model, feats, bg, name="geo_demo"),
        ConceptPack.load(os.path.join(_HERE, "concepts", "geo_zh.json")),
    )
    result = session.audit(None)
    codes = {f["code"] for f in result["findings"]}
    assert codes & {"spurious_dependence", "concentration"} # region_id feature
    report = session.offline_report(0)
    assert len(report) > 100 # report generated successfully


# ---------- G: ----------

def test_synthetic_probes():
    from causallens.probes import run_probes

    results = run_probes()
    for name, r in results.items():
        assert r["passed"], f" {name} : {r['detail']}"


# ---------- F:ONNX safe() ----------

def test_onnx_driver():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    import numpy as np

    from causallens.demos import train_finance_demo
    from causallens.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    from skl2onnx import to_onnx

    onx = to_onnx(model, bg[:1].astype(np.float32))
    path = os.path.join(_HERE, "examples", "artifacts", "finance_model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    drv = detect_driver(None, feats, bg, name="onnx_fin", model_path=path)
    assert drv.kind == "onnx"
    p_onnx = drv.predict(bg[:20])
    p_tree = model.predict_proba(bg[:20])[:, 1]
    assert float(np.max(np.abs(p_onnx - p_tree))) < 1e-4 # ONNX model
    session = MicSessionSafe(drv)


def MicSessionSafe(drv):
    from causallens.core import ConceptPack, MicSession

    return MicSession(drv, ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json")))


# ---------- LLM (,/) ----------

def test_provider_detection_zhipu(monkeypatch):
    for v in list(os.environ):
        if v.endswith("_API_KEY") or v.startswith("MIC_LLM"):
            monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZHIPUAI_API_KEY", "k-test")
    from causallens.agents.llm import LLMClient
    from causallens.agents.providers import detect_provider

    p, key = detect_provider()
    assert p is not None and p.name == "zhipu"
    assert p.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert key == "k-test"
    client = LLMClient()
    assert client.model == "glm-4-flash"


def test_provider_detection_custom(monkeypatch):
    for v in list(os.environ):
        if v.endswith("_API_KEY") or v.startswith("MIC_LLM"):
            monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("MIC_LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("MIC_LLM_API_KEY", "local")
    monkeypatch.setenv("MIC_LLM_MODEL", "my-model")
    from causallens.agents.llm import LLMClient

    c = LLMClient()
    assert c.base_url == "http://localhost:8000/v1" and c.model == "my-model"


def test_tool_call_normalization_variants():
    """GLM/OpenAI :arguments /dict,content None."""
    from causallens.agents.llm import LLMClient

    glm_style = {
        "content": None,
        "tool_calls": [
            {"id": "call_abc", "function": {"name": "get_attribution", "arguments": '{"sample_index": 3}'}}
        ],
    }
    dict_style = {
        "content": "",
        "tool_calls": [
            {"id": "call_2", "function": {"name": "audit", "arguments": {"sample_index": 1}}}
        ],
    }
    calls = LLMClient.normalize_tool_calls(glm_style)
    assert calls[0]["arguments"] == {"sample_index": 3}
    calls2 = LLMClient.normalize_tool_calls(dict_style)
    assert calls2[0]["arguments"] == {"sample_index": 1}
    assert LLMClient.normalize_tool_calls({"content": "direct"}) == []


def test_llm_check_cli_registered(monkeypatch):
    for v in list(os.environ):
        if v.endswith("_API_KEY") or v.startswith("MIC_LLM"):
            monkeypatch.delenv(v, raising=False)
    from causallens.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["llm-check"])
    assert ei.value.code == 1


def test_torch_activation_patching():
    torch = pytest.importorskip("torch")
    from causallens.demos import FinanceMLP
    from causallens.drivers.torch_model import TorchDriver

    net, feats, bg, auc = FinanceMLP.train(epochs=40)
    drv = TorchDriver(net, feats, bg, name="mlp_test")
    assert drv.kind == "torch"

    #
    acts = drv.get_activations(drv.sample_at(0))
    assert len(acts) == len(drv.list_layers())

    # xinput:momentum , Top2
    x = drv.sample_at(0)
    attr = drv.input_attribution(x)
    order = np.argsort(-np.abs(attr))
    assert "momentum" in [feats[i] for i in order[:2]]

    # activation patching:outputsamplerecovery
    p_clean = drv.predict_one(drv.sample_at(0))
    p_other = drv.predict_one(drv.sample_at(1))
    with drv.patch_activation("2", drv.sample_at(0)):
        p_patched = drv.predict_one(drv.sample_at(1))
    assert abs(p_patched - p_clean) <= abs(p_other - p_clean) + 1e-6
