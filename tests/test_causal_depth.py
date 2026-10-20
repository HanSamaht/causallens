"""Causal judgment depth: CATE, double ML, omitted variable."""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge

from intervense.analysis.cate_heterogeneity import cate_heterogeneity
from intervense.analysis.double_ml import double_ml_effect
from intervense.analysis.omitted_variable import omitted_variable_sensitivity
from intervense.core import MicSession
from intervense.drivers import detect_driver


def make_heterogeneous(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    # effect of x1 is 2.0 for x2 > 0 and 0.2 for x2 <= 0
    effect = np.where(x2 > 0, 2.0, 0.2)
    y = effect * x1 + 0.5 * x3 + 0.1 * rng.normal(size=n)
    X = np.column_stack([x1, x2, x3])
    return X, y, ["x1", "x2", "x3"]


def make_confounded(n=1200, seed=1, theta=0.8):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    T = 0.9 * x1 + 0.6 * x2 + rng.normal(scale=0.5, size=n)
    y = theta * T + 1.2 * x1 - 0.8 * x2 + 0.4 * x3 + rng.normal(scale=0.3, size=n)
    X = np.column_stack([T, x1, x2, x3])
    return X, y, ["t", "x1", "x2", "x3"], theta


def test_cate_detects_heterogeneity():
    X, y, feats = make_heterogeneous()
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="cate")
    out = cate_heterogeneity(drv, "x1", y, n_bootstrap=20, seed=0)
    assert out["verdict"].startswith("HETEROGENEOUS"), out["verdict"]
    assert out["top_split_feature"] == "x2"
    # the strong subgroup mean should sit near 2.0, the weak near 0.2
    assert out["subgroups"][0]["mean_effect"] > 1.2
    assert out["subgroups"][-1]["mean_effect"] < 0.7


def test_cate_homogeneous():
    rng = np.random.default_rng(2)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 1.0 * x1 + 0.3 * rng.normal(size=n)
    X = np.column_stack([x1, x2])
    lin = LinearRegression().fit(X, y)
    drv = detect_driver(lin, ["x1", "x2"], X, name="homo")
    out = cate_heterogeneity(drv, "x1", y, n_bootstrap=15, seed=0)
    assert out["verdict"].startswith("HOMOGENEOUS"), out["verdict"]
    # per-row effects are p10->p90 contrasts, about 2.5 SD wide here
    width = out["contrast"]["to"] - out["contrast"]["from"]
    assert abs(out["mean_effect"] / width - 1.0) < 0.15


def test_double_ml_recovers_confounded_effect():
    X, y, feats, theta = make_confounded()
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=2,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="dml")
    out = double_ml_effect(drv, "t", y, n_folds=5, seed=0)
    assert "SIGNIFICANT" in out["verdict"]
    lo, hi = out["ci95"]
    assert lo <= theta <= hi, (lo, hi, theta)
    assert abs(out["theta_per_unit"] - theta) < 0.1
    # the naive uncontrolled estimate should be visibly biased
    assert abs(out["naive_ols_uncontrolled"] - theta) > 0.2


def test_double_ml_null_treatment():
    rng = np.random.default_rng(3)
    n = 800
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    T = 0.8 * x1 + rng.normal(scale=0.5, size=n)
    y = 1.0 * x1 - 0.5 * x2 + rng.normal(scale=0.3, size=n)  # no T effect
    X = np.column_stack([T, x1, x2])
    gbm = GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, ["t", "x1", "x2"], X, name="null")
    out = double_ml_effect(drv, "t", y, n_folds=5, seed=0)
    assert "NOT SIGNIFICANT" in out["verdict"]


def test_omitted_variable_robust_case():
    X, y, feats, _ = make_confounded(n=900, seed=4)
    # all confounders included: delta should clear 1
    lin = Ridge(alpha=0.1).fit(X, y)
    drv = detect_driver(lin, feats, X, name="ovs_r")
    out = omitted_variable_sensitivity(drv, "t", y)
    assert out["verdict"].startswith("ROBUST"), out["verdict"]
    assert out["delta"] > 1


def test_omitted_variable_fragile_case():
    # Oster's actual fragility logic: the confounder is OBSERVED and
    # controlled, but controlling it moves the coefficient a lot.
    # Strong selection on observables is evidence that more selection
    # of the same character could be hiding in the unobservables,
    # so delta lands below 1.
    rng = np.random.default_rng(5)
    n = 900
    x1 = rng.normal(size=n)                      # strong confounder
    T = 1.5 * x1 + rng.normal(scale=0.5, size=n)  # tightly coupled
    y = 0.8 * T + 1.5 * x1 + rng.normal(scale=1.5, size=n)
    X = np.column_stack([T, x1])
    lin = Ridge(alpha=0.1).fit(X, y)
    drv = detect_driver(lin, ["t", "x1"], X, name="ovs_f")
    out = omitted_variable_sensitivity(drv, "t", y)
    assert out["verdict"].startswith("FRAGILE"), out["verdict"]
    assert 0 < out["delta"] < 1


def test_session_and_agent_wiring():
    X, y, feats, theta = make_confounded(n=700, seed=6)
    gbm = GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                    random_state=0).fit(X, y)
    session = MicSession(detect_driver(gbm, feats, X, name="wire"))
    out = session.double_ml_effect("t", y, n_folds=3, seed=0)
    assert "SIGNIFICANT" in out["verdict"]
    assert session.evidence_log[-1]["tool"] == "double_ml_effect"

    from intervense.agents.interrogator import Interrogator

    session.register_data("y", y.tolist())
    result = Interrogator(session)._execute_tool(
        "double_ml_effect", {"treatment": "t"})
    assert "verdict" in result
