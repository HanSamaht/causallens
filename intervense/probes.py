"""Synthetic probes: ground-truth self-validation of the framework.

The probe models are built with the answer printed on the box. If the
framework gets attribution, counterfactuals, or interventions wrong
here, the bug is in the framework, not the model. This follows the
synthetic vision neuron idea from MAIA.

Probes:
  1. linear probe: a known logit, attribution must match the coefficients
  2. leak probe: a planted leakage feature, the auditor must flag it
  3. interaction probe: y = x1 times x2, the causal graph must find it

Run: python -m intervense probe
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
    """Make pack."""
    return ConceptPack.from_dict(
        {
            "domain": "probe",
            "features": {k: {"concept": v, "role": "normal"} for k, v in names.items()},
        }
    )


def probe_linear() -> tuple[bool, str]:
    """Probe linear."""
    coef = {"x1": 1.8, "x2": -1.3, "x3": 0.9, "x4": -0.4, "x5": 0.1}
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    logit = sum(c * X[:, i] for i, c in enumerate(coef.values())) + 0.2 * rng.normal(size=3000)
    y = (logit > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(X, y)
    pack = _make_pack({k: f"feature{i+1}" for i, k in enumerate(coef)})
    session = MicSession(detect_driver(model, list(coef), X[2500:], name="probe_linear"), pack)

    # 1:attribution vs || ,Spearman ≥ 0.8
    sm = np.abs(np.array([a.shap_value for a in explain_sample(session.driver, X[0], pack)] * 1))
    from intervense.analysis.attribution import shap_matrix

    mean_abs = np.abs(shap_matrix(session.driver, X[:300])).mean(axis=0)
    rho = _spearman(mean_abs, np.abs(list(coef.values())))
    if rho is None or rho < 0.8:
        return False, f"attribution ranking distortion:Spearman={rho}(≥0.8)"

    # 2:counterfactual,iseffectfeature
    idx = int(np.argmax(session.driver.predict(X[:300])))
    paths = search_counterfactuals(session.driver, X[idx])
    if not paths:
        return False, "counterfactualto"
    return True, f"Spearman={rho:.3f}; counterfactual {[list(p.changed_features) for p in paths[:2]]}"


def probe_leak() -> tuple[bool, str]:
    """Probe leak."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 4))
    leak = rng.normal(size=3000)
    Xa = np.column_stack([X, leak])
    logit = 0.5 * X[:, 0] + 3.0 * leak + 0.2 * rng.normal(size=3000) # leakagemost
    y = (logit > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(Xa, y)
    pack = ConceptPack.from_dict(
        {
            "domain": "probe",
            "features": {
                "f1": {"concept": "normalfeature", "role": "normal"},
                "f2": {"concept": "normalfeature", "role": "normal"},
                "f3": {"concept": "normalfeature", "role": "normal"},
                "f4": {"concept": "normalfeature", "role": "normal"},
                "leak": {"concept": "leakagefeature", "role": "spurious"},
            },
        }
    )
    session = MicSession(detect_driver(model, ["f1", "f2", "f3", "f4", "leak"], Xa[2500:], name="probe_leak"), pack)
    result = session.audit(None)
    codes = {f["code"] for f in result["findings"]}
    if not codes & {"spurious_dependence", "proxy_dependence", "concentration"}:
        return False, f"auditorleakagefeature;findings={result['findings']}"
    share = result["global_attribution_share"].get("leakagefeature")
    return True, f"auditortoleakagefeature(attribution {share})"


def probe_interact() -> tuple[bool, str]:
    """Probe interact."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(3000, 4))
    logit = 2.0 * X[:, 0] * X[:, 1] + 0.1 * X[:, 2] + 0.2 * rng.normal(size=3000) #
    y = (logit > 0).astype(int)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=0).fit(X, y)
    pack = _make_pack({f"x{i}": f"feature{i}" for i in range(1, 5)})
    session = MicSession(detect_driver(model, ["x1", "x2", "x3", "x4"], X[2500:], name="probe_interact"), pack)
    g = build_causal_graph(session.driver, np.median(session.driver.background, axis=0))
    interact_nodes = {e.source for e in g.interaction_edges} | {e.target for e in g.interaction_edges}
    if not ({"x1", "x2"} & interact_nodes):
        return False, f"by:interaction_edges={[e.to_dict() for e in g.interaction_edges]}"
    return True, f"bycausal graph({len(g.interaction_edges)} intervention)"


PROBES = {"linear": probe_linear, "leak": probe_leak, "interact": probe_interact}


def run_probes() -> dict:
    """Run probes."""
    results = {}
    for name, fn in PROBES.items():
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f": {type(e).__name__}: {e}"
        results[name] = {"passed": ok, "detail": detail}
    return results
