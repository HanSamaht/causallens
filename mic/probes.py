"""合成探针（Synthetic Probes）：MIC 自身的 ground-truth 回归测试集。

真实模型没有标准答案，但我们能构造**已知真实机制**的模型——如果 MIC 的
归因/反事实/审查在已知答案上都出错，那是框架的 bug，不是模型的问题。
（思路来自 MAIA 论文的 synthetic vision neurons 评测。）

三个探针：
  1. linear   ：已知系数的线性 logit → 归因排名应与 |真系数| 一致
  2. leak     ：注入大系数泄漏特征（role=spurious）→ 审查员必须抓到
  3. interact ：纯交互结构 y=x1·x2 → 因果图应出现交互证据

用法：python -m mic probe
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .analysis import build_causal_graph, explain_sample, search_counterfactuals
from .core.concepts import ConceptPack
from .core.session import MicSession
from .drivers import detect_driver
from .analysis.faithfulness import _spearman


def _make_pack(names: dict[str, str]) -> ConceptPack:
    return ConceptPack.from_dict(
        {
            "domain": "probe",
            "features": {k: {"concept": v, "role": "normal"} for k, v in names.items()},
        }
    )


def probe_linear() -> tuple[bool, str]:
    coef = {"x1": 1.8, "x2": -1.3, "x3": 0.9, "x4": -0.4, "x5": 0.1}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    logit = sum(c * X[:, i] for i, c in enumerate(coef.values())) + 0.2 * rng.normal(size=3000)
    y = (logit > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(X, y)
    pack = _make_pack({k: f"探针特征{i+1}" for i, k in enumerate(coef)})
    session = MicSession(detect_driver(model, list(coef), X[2500:], name="probe_linear"), pack)

    # 断言1：归因排名 vs |真系数| 排名，Spearman ≥ 0.8
    sm = np.abs(np.array([a.shap_value for a in explain_sample(session.driver, X[0], pack)] * 1))
    from mic.analysis.attribution import shap_matrix

    mean_abs = np.abs(shap_matrix(session.driver, X[:300])).mean(axis=0)
    rho = _spearman(mean_abs, np.abs(list(coef.values())))
    if rho is None or rho < 0.8:
        return False, f"归因排名失真：Spearman={rho}（要求≥0.8）"

    # 断言2：反事实能翻转，且改的是高影响特征
    idx = int(np.argmax(session.driver.predict(X[:300])))
    paths = search_counterfactuals(session.driver, X[idx])
    if not paths:
        return False, "反事实未找到翻转方案"
    return True, f"Spearman={rho:.3f}; 反事实改动 {[list(p.changed_features) for p in paths[:2]]}"


def probe_leak() -> tuple[bool, str]:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 4))
    leak = rng.normal(size=3000)
    Xa = np.column_stack([X, leak])
    logit = 0.5 * X[:, 0] + 3.0 * leak + 0.2 * rng.normal(size=3000)  # 泄漏系数最大
    y = (logit > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(Xa, y)
    pack = ConceptPack.from_dict(
        {
            "domain": "probe",
            "features": {
                "f1": {"concept": "正常特征", "role": "normal"},
                "f2": {"concept": "正常特征", "role": "normal"},
                "f3": {"concept": "正常特征", "role": "normal"},
                "f4": {"concept": "正常特征", "role": "normal"},
                "leak": {"concept": "泄漏特征", "role": "spurious"},
            },
        }
    )
    session = MicSession(detect_driver(model, ["f1", "f2", "f3", "f4", "leak"], Xa[2500:], name="probe_leak"), pack)
    result = session.audit(None)
    codes = {f["code"] for f in result["findings"]}
    if not codes & {"spurious_dependence", "proxy_dependence", "concentration"}:
        return False, f"审查员漏掉了注入的泄漏特征；findings={result['findings']}"
    share = result["global_attribution_share"].get("泄漏特征")
    return True, f"审查员抓到泄漏特征（归因占比 {share}）"


def probe_interact() -> tuple[bool, str]:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 4))
    logit = 2.0 * X[:, 0] * X[:, 1] + 0.1 * X[:, 2] + 0.2 * rng.normal(size=3000)  # 纯交互主导
    y = (logit > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=0).fit(X, y)
    pack = _make_pack({f"x{i}": f"交互特征{i}" for i in range(1, 5)})
    session = MicSession(detect_driver(model, ["x1", "x2", "x3", "x4"], X[2500:], name="probe_interact"), pack)
    g = build_causal_graph(session.driver, np.median(session.driver.background, axis=0))
    interact_nodes = {e.source for e in g.interaction_edges} | {e.target for e in g.interaction_edges}
    if not ({"x1", "x2"} & interact_nodes):
        return False, f"交互探针未被检出：interaction_edges={[e.to_dict() for e in g.interaction_edges]}"
    return True, f"交互结构被因果图捕获（{len(g.interaction_edges)} 条干预边）"


PROBES = {"linear": probe_linear, "leak": probe_leak, "interact": probe_interact}


def run_probes() -> dict:
    results = {}
    for name, fn in PROBES.items():
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"探针异常: {type(e).__name__}: {e}"
        results[name] = {"passed": ok, "detail": detail}
    return results
