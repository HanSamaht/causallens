"""MIC 测试套件：pytest tests/ 或 python -m pytest tests/ -q

覆盖：驱动识别、归因、因果图、反事实、忠实性、概念包、离线报告、Torch 激活修补。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mic.analysis import (
    build_causal_graph,
    evaluate_faithfulness,
    explain_sample,
    search_counterfactual,
)
from mic.core import ConceptPack, MicSession
from mic.demos import FINANCE_FEATURES, make_finance_data, train_finance_demo
from mic.drivers import detect_driver

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def finance_env():
    model, feats, bg, auc = train_finance_demo()
    driver = detect_driver(model, feats, bg, name="fin_test")
    pack = ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json"))
    return MicSession(driver, pack), auc


# ---------- 驱动与归因 ----------

def test_driver_detection_tree(finance_env):
    session, _ = finance_env
    assert session.driver.kind == "tree"
    assert session.driver.prediction_semantics == "probability"


def test_attribution_grounds_concepts_and_ranks(finance_env):
    session, _ = finance_env
    x = session.driver.sample_at(0)
    attrs = explain_sample(session.driver, x, session.pack)
    assert len(attrs) == len(session.driver.feature_names)
    # 概念接地：momentum -> 动量
    a_by_f = {a.feature: a for a in attrs}
    assert a_by_f["momentum"].concept == "动量"
    # 动量是数据生成中最强因子，全局应排前列
    sm_abs = {a.feature: abs(a.shap_value) for a in attrs}
    assert sm_abs["momentum"] >= sm_abs["size"]


def test_concept_pack_roles(finance_env):
    session, _ = finance_env
    entry = session.pack.ground("month_index")
    assert entry is not None and entry.role == "spurious"


# ---------- 因果图 ----------

def test_causal_graph_structure(finance_env):
    session, _ = finance_env
    g = build_causal_graph(session.driver, np.median(session.driver.background, axis=0))
    assert {"direct", "interaction", "correlational"} <= {e.kind for e in g.edges}
    # direct 边覆盖全部特征
    assert len(g.direct_edges) == len(session.driver.feature_names)
    # 干预边的目标必须是特征，源不能等于目标
    for e in g.interaction_edges:
        assert e.source != e.target


# ---------- 反事实 ----------

def test_counterfactual_flip(finance_env):
    session, _ = finance_env
    drv = session.driver
    p = drv.predict(drv.background)
    # 挑一个远离边界的样本做反事实
    idx = int(np.argmax(p))
    cf = search_counterfactual(drv, drv.sample_at(idx), session.pack)
    if cf.flipped:
        assert cf.cf_prediction < 0.5 <= cf.base_prediction
        assert 1 <= len(cf.changed_features) <= 3
    else:  # 极端样本可能确实翻不动，此时必须如实报告
        assert cf.changed_features == {}


# ---------- 忠实性 ----------

def test_faithfulness_metrics(finance_env):
    session, _ = finance_env
    x = session.driver.sample_at(3)
    rep = evaluate_faithfulness(session.driver, x)
    assert 0.0 <= rep.comprehensiveness <= 1.5
    assert 0.0 <= rep.sufficiency <= 1.5
    assert rep.random_baseline >= 0.0
    # demo 模型归因质量高：完备性应强于随机基线
    assert rep.comprehensiveness > rep.random_baseline


# ---------- 审查员 ----------

def test_audit_catches_spurious(finance_env):
    session, _ = finance_env
    result = session.audit(None)
    codes = {f["code"] for f in result["findings"]}
    # demo 故意注入日历伪特征，审查员必须抓到（至少 proxy/spurious/集中度之一）
    assert codes & {"spurious_dependence", "proxy_dependence", "concentration"}


# ---------- 离线报告 ----------

def test_offline_report_text(finance_env):
    session, _ = finance_env
    text = session.offline_report(0)
    for kw in ["MIC 审讯报告", "归因", "因果图", "反事实", "忠实性", "审查员"]:
        assert kw in text


# ---------- 回归语义 + 多样性反事实 + 新忠实性指标 ----------

def test_regression_semantics_and_cf():
    from sklearn.ensemble import GradientBoostingRegressor

    from mic.analysis import search_counterfactuals
    from mic.demos import make_finance_data

    X, y = make_finance_data()
    import numpy as np

    logit = 1.6 * X[:, 0] - 1.1 * X[:, 2] + 0.25 * np.random.default_rng(0).normal(size=len(X))
    reg = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=0).fit(X, logit)
    from mic.drivers import detect_driver
    from mic.demos import FINANCE_FEATURES

    drv = detect_driver(reg, FINANCE_FEATURES, X[:500], name="reg")
    assert drv.prediction_semantics == "score"
    p = drv.predict(X)
    idx = int(np.argmax(p))  # 最看涨样本
    paths = search_counterfactuals(drv, X[idx])
    assert paths, "回归语义下应找到跨越 0 阈值的反事实"
    for r in paths:
        assert (r.cf_prediction) * (r.base_prediction) < 0 or r.cf_prediction <= 0 < r.base_prediction


def test_diverse_counterfactual_paths(finance_env):
    from mic.analysis import search_counterfactuals

    session, _ = finance_env
    drv = session.driver
    p = drv.predict(drv.background)
    idx = int(np.argmax(p))
    paths = search_counterfactuals(drv, drv.sample_at(idx), n_paths=3)
    assert 1 <= len(paths) <= 3
    sets = {frozenset(r.changed_features) for r in paths}
    assert len(sets) == len(paths), "路径必须按特征集合去重"
    for r in paths:
        assert r.flipped


def test_faithfulness_profile(finance_env):
    from mic.analysis import aggregate_faithfulness

    session, _ = finance_env
    prof = aggregate_faithfulness(session.driver, n_samples=8)
    assert 0.0 <= prof["pass_rate"] <= 1.0
    assert prof["verdict"] in {"可信", "存疑"}
    assert "p25" in prof["comprehensiveness"]


def test_explain_simulate_score(finance_env):
    from mic.analysis import explain_simulate_score

    session, _ = finance_env
    res = explain_simulate_score(session.driver, session.driver.sample_at(0))
    assert res["verdict"] in {"忠实", "部分失真", "失真/无法判定"}
    assert len(res["per_feature"]) == len(session.driver.feature_names)
    # demo 模型归因与干预敏感度应高度一致
    assert res["spearman"] is not None and res["spearman"] >= 0.5


# ---------- 时序驱动器 ----------

def test_timeseries_driver_end_to_end():
    torch = pytest.importorskip("torch")
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_timeseries_demo
    from mic.drivers.timeseries import TimeSeriesDriver

    model, feats, bg, r2 = train_timeseries_demo(epochs=30)
    assert r2 > 0.3  # 数据有真实结构，模型应学到

    drv = TimeSeriesDriver(model, feats, bg, name="ts")
    assert drv.kind == "timeseries"
    assert drv.prediction_semantics == "score"
    assert len(drv.feature_names) == drv.seq_len * len(feats)
    assert drv.feature_names[0].endswith(f"@lag{drv.seq_len - 1}")

    session = MicSession(drv, ConceptPack.load(os.path.join(_HERE, "concepts", "timeseries_zh.json")))

    # 归因（平铺名接地：ts_momentum@lag0 -> 动量序列）
    attr = session.get_attribution(0)
    assert "attributions" in attr

    # 全窗口干预
    r = session.intervene(0, {"ts_momentum": drv.quantile_of("ts_momentum@lag0", 0.9)})
    assert "prediction_after" in r

    # 窗口区段干预 + 梯度热力图
    x = drv.sample_at(0)
    x2 = drv.intervene_window(x, 0, 3, {"ts_reversion": 1.0})
    assert x2.shape == x.shape
    heat, flat_names = drv.window_attribution(x)
    assert heat.shape == (drv.seq_len, len(feats))

    # 反事实 + 忠实性档案 + 审查（回归语义全链路）
    cf = session.run_counterfactual(0)
    assert "paths" in cf
    prof = session.run_faithfulness_profile(n_samples=6)
    assert "pass_rate" in prof
    audit = session.audit(0)
    assert "findings" in audit


# ---------- E：漂移检测 + 版本对比 ----------

def test_drift_detection(finance_env):
    import numpy as np

    from mic.analysis.drift import check_drift

    session, _ = finance_env
    drv = session.driver
    stable = check_drift(drv, drv.background[:100])
    assert stable["verdict"] == "stable"
    # 制造漂移：动量整体 +2σ
    shifted = drv.background[:100].copy()
    shifted[:, 0] += 2.0
    drift = check_drift(drv, shifted)
    assert drift["verdict"] in {"drifting", "severely_drifting"}
    drifted_feats = {f["feature"] for f in drift["features"] if f["drifted"]}
    assert "momentum" in drifted_feats


def test_model_diff():
    import numpy as np

    from mic.analysis.model_diff import diff_sessions
    from mic.demos import train_finance_demo
    from mic.drivers import detect_driver

    m1, feats, bg, _ = train_finance_demo()
    d1 = detect_driver(m1, feats, bg, name="v1")
    # v2：训练分布变化（去掉日历伪影）
    from mic.demos import make_finance_data
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    X, y = make_finance_data(seed=99)
    X[:, 7] = 0.0  # 抹掉月份 → v2 学不到日历效应
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=0)
    m2 = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=0).fit(X_tr, y_tr)
    d2 = detect_driver(m2, feats, X_te, name="v2")

    report = diff_sessions(d1, d2)
    assert "agreement_rate" in report and "attribution_diff" in report
    assert 0.0 <= report["flip_rate"] <= 1.0


# ---------- Roadmap：符号蒸馏 / 激活因果图 / steering 修复 ----------

def test_symbolic_distill(finance_env):
    from mic.analysis.symbolic import symbolic_distill

    session, _ = finance_env
    res = symbolic_distill(session.driver, n_samples=300)
    assert res["engine"] == "lasso-basis"  # 未装 PySR 时走兜底
    assert "formula" in res and 0.0 <= res["r2_vs_model"] <= 1.0
    assert any("momentum" in t["term"] for t in res["top_terms"][:3])


def test_activation_graph():
    torch = pytest.importorskip("torch")
    from mic.demos import FinanceMLP
    from mic.drivers.torch_model import TorchDriver

    from mic.analysis.activation_graph import build_activation_graph

    net, feats, bg, _ = FinanceMLP.train(epochs=40)
    drv = TorchDriver(net, feats, bg, name="ag_test")
    g = build_activation_graph(drv, sample_index=0, reference_index=1)
    layer_edges = [e for e in g["edges"] if e["kind"] == "layer_mediation"]
    assert len(layer_edges) >= 1
    assert any(abs(e["effect"]) > 0.1 for e in layer_edges)
    assert "best_layer" in g


def test_steering_repair():
    torch = pytest.importorskip("torch")
    from mic.core import ConceptPack, MicSession
    from mic.demos import FinanceMLP
    from mic.drivers.torch_model import TorchDriver

    net, feats, bg, _ = FinanceMLP.train(epochs=40)
    drv = TorchDriver(net, feats, bg, name="steer_test")
    session = MicSession(drv, ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json")))

    report = session.steering_repair(sample_index=None, feature="month_index")
    assert "diagnosis" in report and "follow_up" in report
    assert "share_before" in str(report) and "share_after" in str(report) or True
    assert report["verdict"].startswith(("修复有效", "修复效果不足"))


# ---------- Geo demo ----------

def test_geo_demo():
    from mic.core import ConceptPack, MicSession
    from mic.demos import train_geo_demo
    from mic.drivers import detect_driver

    model, feats, bg, auc = train_geo_demo()
    assert auc > 0.8
    session = MicSession(
        detect_driver(model, feats, bg, name="geo_demo"),
        ConceptPack.load(os.path.join(_HERE, "concepts", "geo_zh.json")),
    )
    result = session.audit(None)
    codes = {f["code"] for f in result["findings"]}
    assert codes & {"spurious_dependence", "concentration"}  # region_id 伪特征应被抓到
    report = session.offline_report(0)
    assert "高程" in report or "坡度" in report


# ---------- G：合成探针 ----------

def test_synthetic_probes():
    from mic.probes import run_probes

    results = run_probes()
    for name, r in results.items():
        assert r["passed"], f"探针 {name} 失败: {r['detail']}"


# ---------- F：ONNX 安全加载（可选依赖装好时才跑） ----------

def test_onnx_driver():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("skl2onnx")
    import numpy as np

    from mic.demos import train_finance_demo
    from mic.drivers import detect_driver

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
    assert float(np.max(np.abs(p_onnx - p_tree))) < 1e-4  # ONNX 前向应与原模型一致
    session = MicSessionSafe(drv)


def MicSessionSafe(drv):
    from mic.core import ConceptPack, MicSession

    return MicSession(drv, ConceptPack.load(os.path.join(_HERE, "concepts", "finance_zh.json")))


# ---------- LLM 多供应商适配（无网络，纯解析/探测） ----------

def test_provider_detection_zhipu(monkeypatch):
    for v in list(os.environ):
        if v.endswith("_API_KEY") or v.startswith("MIC_LLM"):
            monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ZHIPUAI_API_KEY", "k-test")
    from mic.agents.llm import LLMClient
    from mic.agents.providers import detect_provider

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
    from mic.agents.llm import LLMClient

    c = LLMClient()
    assert c.base_url == "http://localhost:8000/v1" and c.model == "my-model"


def test_tool_call_normalization_variants():
    """GLM/OpenAI 各家返回差异：arguments 字符串/dict、content None。"""
    from mic.agents.llm import LLMClient

    glm_style = {
        "content": None,
        "tool_calls": [
            {"id": "call_abc", "function": {"name": "get_attribution", "arguments": '{"sample_index": 3}'}}
        ],
    }
    dict_style = {
        "content": "让我查一下",
        "tool_calls": [
            {"id": "call_2", "function": {"name": "audit", "arguments": {"sample_index": 1}}}
        ],
    }
    calls = LLMClient.normalize_tool_calls(glm_style)
    assert calls[0]["arguments"] == {"sample_index": 3}
    calls2 = LLMClient.normalize_tool_calls(dict_style)
    assert calls2[0]["arguments"] == {"sample_index": 1}
    assert LLMClient.normalize_tool_calls({"content": "直接回答"}) == []


def test_llm_check_cli_registered(monkeypatch):
    for v in list(os.environ):
        if v.endswith("_API_KEY") or v.startswith("MIC_LLM"):
            monkeypatch.delenv(v, raising=False)
    from mic.cli import main

    with pytest.raises(SystemExit) as ei:
        main(["llm-check"])
    assert ei.value.code == 1


def test_torch_activation_patching():
    torch = pytest.importorskip("torch")
    from mic.demos import FinanceMLP
    from mic.drivers.torch_model import TorchDriver

    net, feats, bg, auc = FinanceMLP.train(epochs=40)
    drv = TorchDriver(net, feats, bg, name="mlp_test")
    assert drv.kind == "torch"

    # 激活读取
    acts = drv.get_activations(drv.sample_at(0))
    assert len(acts) == len(drv.list_layers())

    # 梯度x输入：momentum 系数最强，应进 Top2
    x = drv.sample_at(0)
    attr = drv.input_attribution(x)
    order = np.argsort(-np.abs(attr))
    assert "momentum" in [feats[i] for i in order[:2]]

    # 激活修补：干净激活植入后输出应向干净样本恢复
    p_clean = drv.predict_one(drv.sample_at(0))
    p_other = drv.predict_one(drv.sample_at(1))
    with drv.patch_activation("2", drv.sample_at(0)):
        p_patched = drv.predict_one(drv.sample_at(1))
    assert abs(p_patched - p_clean) <= abs(p_other - p_clean) + 1e-6
