"""Specification audit: sign stability of intervention effects."""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from intervense.analysis.specification_audit import specification_audit
from intervense.core import MicSession
from intervense.drivers import detect_driver


def make_suppressor(n=800, seed=21):
    """x helps, z hurts, both ride the same latent cause.

    Lean fit of y on x alone is negative (the suppressor dominates);
    controlling z recovers the positive direct effect, so the audit
    must flag a flip whose minimal set is exactly {z}.
    """
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    x = 0.8 * c + rng.normal(scale=0.6, size=n)
    z = 0.8 * c + rng.normal(scale=0.6, size=n)
    n1 = rng.normal(size=n)
    n2 = rng.normal(size=n)
    y = 0.3 * x - 1.0 * z + 0.2 * n1 + rng.normal(scale=0.3, size=n)
    X = np.column_stack([x, z, n1, n2])
    return X, y, ["x", "z", "n1", "n2"]


def make_independent(n=600, seed=1):
    """No collinearity: every sign should be stable."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 1.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.2, size=n)
    return X, y, ["a", "b", "c"]


@pytest.fixture(scope="module")
def suppressor_linear():
    X, y, feats = make_suppressor()
    model = LinearRegression().fit(X, y)
    return detect_driver(model, feats, X, name="suppressor_linear"), X, y


@pytest.fixture(scope="module")
def independent_linear():
    X, y, feats = make_independent()
    model = LinearRegression().fit(X, y)
    return detect_driver(model, feats, X, name="independent_linear"), X, y


def test_flip_detected_with_minimal_set(suppressor_linear):
    drv, X, y = suppressor_linear
    out = specification_audit(drv, "x", y, max_flip_depth=2)
    assert out["verdict"].startswith("FRAGILE SIGN")
    assert out["effect_lean"] < 0 < out["effect_full"]
    assert out["engine"] == "closed_form_linear"
    assert out["lattice"] == "exact"
    assert out["minimal_flip_sets"] == [
        {"controls": ["z"], "effect": out["minimal_flip_sets"][0]["effect"]}
    ]
    # every specification containing z flips: 4 of 8
    assert out["n_flip"] == 4
    assert out["n_specs"] == 8


def test_flip_listed_specifications(suppressor_linear):
    drv, X, y = suppressor_linear
    out = specification_audit(drv, "x", y)
    controls = [set(f["controls"]) for f in out["flip_specifications"]]
    assert all("z" in c for c in controls)
    assert len(controls) == out["n_flip"]


def test_attribution_gap_flags_healthy_importance(suppressor_linear):
    drv, X, y = suppressor_linear
    out = specification_audit(drv, "x", y)
    assert out["attribution_share"] is not None and out["attribution_share"] > 0
    assert out["attribution_gap"] and "attribution share" in out["attribution_gap"]


def test_stable_sign_when_independent(independent_linear):
    drv, X, y = independent_linear
    for feat in ["a", "b"]:
        out = specification_audit(drv, feat, y)
        assert out["verdict"].startswith("STABLE SIGN"), out["verdict"]
        assert out["n_flip"] == 0
        assert out["minimal_flip_sets"] == []


def test_negligible_for_noise_feature(suppressor_linear):
    drv, X, y = suppressor_linear
    out = specification_audit(drv, "n2", y)
    assert out["verdict"].startswith("NEGLIGIBLE")


def test_ridge_closed_form_matches_refit():
    X, y, feats = make_suppressor(n=400, seed=2)
    ridge = Ridge(alpha=1.0).fit(X, y)
    drv = detect_driver(ridge, feats, X, name="ridge_cert")
    out = specification_audit(drv, "x", y)
    assert out["engine"] == "closed_form_linear"
    # closed form must agree with the sklearn fit on the full specification
    assert abs(out["effect_full"] - (ridge.coef_[0] * (out["contrast"]["to_value"]
                                                       - out["contrast"]["from_value"]))) < 5e-3


def test_ridgecv_uses_fitted_alpha():
    X, y, feats = make_suppressor(n=400, seed=3)
    rcv = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(X, y)
    drv = detect_driver(rcv, feats, X, name="ridgecv_cert")
    out = specification_audit(drv, "x", y)
    assert out["engine"] == "closed_form_linear"
    assert out["verdict"].startswith("FRAGILE SIGN")


def test_sampled_lattice_agrees_with_exact():
    rng = np.random.default_rng(4)
    n = 600
    c = rng.normal(size=n)
    x = 0.8 * c + rng.normal(scale=0.6, size=n)
    z = 0.8 * c + rng.normal(scale=0.6, size=n)
    fillers = [rng.normal(scale=0.5, size=n) for _ in range(10)]
    X = np.column_stack([x, z] + fillers)
    y = 0.3 * x - 1.0 * z + rng.normal(scale=0.3, size=n)
    feats = ["x", "z"] + [f"f{i}" for i in range(10)]
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, feats, X, name="sampled")

    exact = specification_audit(drv, "x", y)
    sampled = specification_audit(drv, "x", y, exact_threshold=64,
                                  max_specs=200, seed=7)
    assert exact["lattice"] == "exact" and sampled["lattice"] == "sampled"
    # z is present in half the lattice; both engines must see about half flip
    assert abs(exact["frac_flip"] - 0.5) < 0.02
    assert abs(sampled["frac_flip"] - 0.5) < 0.15
    assert "flip_ci95" in sampled and 0.0 <= sampled["flip_ci95"][0] <= 1.0


def test_refit_engine_gradient_boosting():
    X, y, feats = make_suppressor(n=400, seed=5)
    gbm = GradientBoostingRegressor(n_estimators=40, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="gbm_spec")
    out = specification_audit(drv, "x", y)
    assert out["engine"] == "refit_clone"
    assert out["verdict"].startswith(("STABLE SIGN", "FRAGILE SIGN", "NEGLIGIBLE"))
    assert out["n_specs"] == 8


def test_errors(suppressor_linear):
    drv, X, y = suppressor_linear
    with pytest.raises(ValueError, match="unknown treatment"):
        specification_audit(drv, "nope", y)
    with pytest.raises(ValueError, match="aligned with"):
        specification_audit(drv, "x", y[:-1])


def test_fold_stability_detects_heterogeneity():
    """Effect positive in two cohorts, reversed in a third: the pool
    cancels toward zero and the fold block names cohort c."""
    rng = np.random.default_rng(9)
    n = 600
    x = rng.normal(size=n)
    z = 0.5 * x + rng.normal(scale=0.5, size=n)
    groups = np.array(["a"] * 200 + ["b"] * 200 + ["c"] * 200)
    y = np.where(groups == "c", -2.0 * x, x) + 0.2 * rng.normal(size=n)
    X = np.column_stack([x, z])
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, ["x", "z"], X, name="hetero")

    out = specification_audit(drv, "x", y, groups=groups)
    assert "fold_stability" in out
    assert out["fold_stability"]["lean_flip_folds"] == ["c"]
    assert "FOLD FRAGILE" in out["verdict"]
    assert "c" in out["verdict"]
    by_group = {f["group"]: f["effect_lean"] for f in out["fold_stability"]["folds"]}
    assert by_group["a"] > 0 > by_group["c"]


def test_fold_stability_agrees_when_homogeneous():
    rng = np.random.default_rng(10)
    n = 600
    x = rng.normal(size=n)
    z = 0.5 * x + rng.normal(scale=0.5, size=n)
    groups = np.array(["a"] * 300 + ["b"] * 300)
    y = x + 0.2 * rng.normal(size=n)
    X = np.column_stack([x, z])
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, ["x", "z"], X, name="homo")

    out = specification_audit(drv, "x", y, groups=groups)
    assert out["fold_stability"]["lean_flip_folds"] == []
    assert "fold stable" in out["verdict"]


def test_fold_groups_misaligned_raises():
    X, y, feats = make_suppressor(n=400, seed=12)
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, feats, X, name="mis")
    with pytest.raises(ValueError, match="groups"):
        specification_audit(drv, "x", y, groups=np.array(["a"] * 10))


def test_certificate_matches_pipeline_penalty_geometry():
    """The closed form must run in the pipeline's transformed space:
    a StandardScaler plus Ridge pipeline regularizes standardized
    columns, and the certificate has to reproduce that fit, not a
    raw space ridge."""
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    X, y, feats = make_suppressor(n=400, seed=13)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    drv = detect_driver(pipe, feats, X, name="pipe_cert")
    out = specification_audit(drv, "x", y)
    assert out["engine"] == "closed_form_linear"

    m = clone(pipe).fit(X, y)
    lo, hi = np.percentile(X[:, 0], 10), np.percentile(X[:, 0], 90)
    A, B = X.copy(), X.copy()
    A[:, 0] = hi
    B[:, 0] = lo
    manual = float(np.mean(m.predict(A) - m.predict(B)))
    assert abs(out["effect_full"] - manual) < 1e-5


def test_estimator_factory_for_non_cloneable_families():
    """A model sklearn cannot clone must be refused with guidance, and
    an estimator_factory must make the audit work for that family."""
    X, y, feats = make_suppressor(n=400, seed=14)

    class FakeNet:
        pass

    drv = detect_driver(FakeNet(), feats, X, name="fake")
    with pytest.raises(ValueError, match="estimator_factory"):
        specification_audit(drv, "x", y)

    from sklearn.linear_model import Ridge

    out = specification_audit(drv, "x", y,
                              estimator_factory=lambda: Ridge(alpha=1.0))
    assert out["engine"] == "refit_clone"
    assert out["verdict"].startswith("FRAGILE SIGN")
    assert out["minimal_flip_sets"]


def test_session_and_agent_wiring():
    X, y, feats = make_suppressor(n=400, seed=6)
    model = LinearRegression().fit(X, y)
    session = MicSession(detect_driver(model, feats, X, name="wiring"))
    out = session.specification_audit("x", y)
    assert out["verdict"].startswith("FRAGILE SIGN")
    assert session.evidence_log[-1]["tool"] == "specification_audit"

    from intervense.agents.interrogator import Interrogator

    session.register_data("y", y.tolist())
    result = Interrogator(session)._execute_tool(
        "specification_audit", {"treatment": "x"}
    )
    assert "verdict" in result and result["verdict"].startswith("FRAGILE SIGN")
