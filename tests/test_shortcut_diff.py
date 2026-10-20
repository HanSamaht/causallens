"""Shortcut diff: model-implied reliance versus the world graph."""
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from intervense.analysis.discovery import HAS_CAUSAL_LEARN
from intervense.analysis.shortcut_diff import shortcut_diff
from intervense.core import MicSession
from intervense.drivers import detect_driver

pytestmark = pytest.mark.skipif(
    not HAS_CAUSAL_LEARN, reason="causal-learn is not installed"
)


def make_supported(n=600, seed=0):
    """Two independent causes, no spurious features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.3 * rng.normal(size=n)
    return X, y, ["a", "b", "filler"]


def make_spurious(n=800, seed=11):
    """x causes y; s rides the same latent cause; noise is noise."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    x = 0.7 * c + rng.normal(scale=0.7, size=n)
    s = 0.7 * c + rng.normal(scale=0.7, size=n)
    y = 0.6 * x + rng.normal(scale=0.6, size=n)
    noise = rng.normal(size=n)
    return np.column_stack([x, s, noise]), y, ["x", "s", "noise"]


def make_proxy(n=800, seed=7):
    """p is an echo of the outcome (leakage flavored measurement)."""
    rng = np.random.default_rng(seed)
    c = rng.normal(size=n)
    x = 0.8 * c + rng.normal(scale=0.6, size=n)
    s = 0.8 * c + rng.normal(scale=0.6, size=n)
    y = 0.5 * x + rng.normal(scale=0.5, size=n)
    p = y + rng.normal(scale=0.3, size=n)
    noise = rng.normal(size=n)
    return np.column_stack([x, s, p, noise]), y, ["x", "s", "p", "noise"]


@pytest.fixture(scope="module")
def supported_linear():
    X, y, feats = make_supported()
    model = LinearRegression().fit(X, y)
    return detect_driver(model, feats, X, name="supported_linear")


def test_supported_reliance_is_causal(supported_linear):
    X, y, _ = make_supported()
    out = shortcut_diff(supported_linear, y, edges=[("a", "outcome"), ("b", "outcome")])
    by_name = {r["feature"]: r for r in out["reliance"]}
    assert by_name["a"]["classification"] == "CAUSAL PATH"
    assert by_name["b"]["classification"] == "CAUSAL PATH"
    assert out["verdict"].startswith("SUPPORTED")
    assert out["reliance_at_risk"] < 0.1


def test_spurious_feature_flagged():
    X, y, feats = make_spurious()
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="spurious")
    out = shortcut_diff(drv, y, edges=[("x", "outcome")])
    by_name = {r["feature"]: r for r in out["reliance"]}
    assert by_name["x"]["classification"] == "CAUSAL PATH"
    # s connects only through an undirected CPDAG edge, noise not at all
    assert by_name["s"]["classification"] == "UNRESOLVED PATH"
    assert by_name["noise"]["classification"] == "INDEPENDENT"
    assert out["verdict"].startswith("SHORTCUT RISK")
    assert 0.05 < out["reliance_at_risk"] < 0.5


def test_label_echo_classified_proxy():
    X, y, feats = make_proxy()
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="proxy")
    out = shortcut_diff(drv, y, edges=[("x", "outcome"), ("outcome", "p")])
    by_name = {r["feature"]: r for r in out["reliance"]}
    assert by_name["p"]["classification"] == "PROXY"
    assert by_name["x"]["classification"] == "CAUSAL PATH"
    assert out["verdict"].startswith("SHORTCUT RISK")
    assert "proxy" in out["verdict"]


def test_user_edges_pin_orientation():
    """Without user edges the CPDAG may leave x to outcome undirected."""
    X, y, feats = make_supported()
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, feats, X, name="pin")
    pinned = shortcut_diff(drv, y, edges=[("a", "outcome"), ("b", "outcome")])
    directed = {(e["source"], e["target"]) for e in pinned["world_edges"]
                if e["directed"]}
    assert ("a", "outcome") in directed and ("b", "outcome") in directed


def test_nan_input_refused(supported_linear):
    X, y, _ = make_supported()
    bad = X.copy()
    bad[0, 0] = np.nan
    drv = detect_driver(LinearRegression().fit(X, y), ["a", "b", "filler"],
                        bad, name="nan_drv")
    with pytest.raises(ValueError, match="impute"):
        shortcut_diff(drv, y)


def test_misaligned_y_refused(supported_linear):
    X, y, _ = make_supported()
    with pytest.raises(ValueError, match="aligned"):
        shortcut_diff(supported_linear, y[:-1])


def test_lingam_backend_runs():
    X, y, feats = make_spurious()
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="lingam_drv")
    out = shortcut_diff(drv, y, method="lingam")
    assert out["discovery_method"] == "lingam"
    assert all(e["directed"] for e in out["world_edges"])
    assert out["verdict"].startswith(("SUPPORTED", "SHORTCUT RISK"))


def test_permutation_null_separates_signal_from_chance():
    """The proxy share must exceed the refit-on-shuffled-y baseline."""
    X, y, feats = make_proxy()
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="perm")
    out = shortcut_diff(drv, y, edges=[("x", "outcome"), ("outcome", "p")],
                        n_perm=20)
    by_name = {r["feature"]: r for r in out["reliance"]}
    assert "perm_p" in by_name["p"]
    # 20 replicates: the smallest reachable p is 1/21 (rounded to 0.048)
    assert by_name["p"]["perm_p"] < 0.05
    assert by_name["p"]["reliance_share"] > by_name["p"]["null_share_median"]
    assert out["permutation"] == {"n_perm": 20}
    assert "perm p=" in out["verdict"]


def test_permutation_disabled():
    X, y, feats = make_spurious()
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="noperm")
    out = shortcut_diff(drv, y, edges=[("x", "outcome")], n_perm=0)
    assert out["permutation"] is None
    assert all("perm_p" not in r for r in out["reliance"])


def test_unknown_edge_name_raises():
    """A typo in a user edge must refuse, not silently skip."""
    X, y, feats = make_spurious()
    gbm = GradientBoostingRegressor(n_estimators=40, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="typo")
    with pytest.raises(ValueError, match="unknown feature"):
        shortcut_diff(drv, y, edges=[("x_typo", "outcome")])
    with pytest.raises(ValueError, match="latent variables"):
        shortcut_diff(drv, y, edges=[("unobserved_cause", "x")])


def test_graph_bootstrap_stability():
    """Pinned user edges make classifications stable across graph
    re-estimation; the class_prob field reports that stability."""
    X, y, feats = make_proxy()
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="gboot")
    out = shortcut_diff(drv, y, edges=[("x", "outcome"), ("outcome", "p")],
                        n_perm=0, n_graph_boot=8)
    by_name = {r["feature"]: r for r in out["reliance"]}
    assert by_name["p"]["class_prob"] == 1.0
    assert by_name["x"]["class_prob"] == 1.0
    assert out["graph_bootstrap"]["n"] == 8

    # without pinned edges the orientation is data driven, probabilities
    # must still be present and calibrated in [0, 1]
    out2 = shortcut_diff(drv, y, n_perm=0, n_graph_boot=8)
    for r in out2["reliance"]:
        if "class_prob" in r:
            assert 0.0 <= r["class_prob"] <= 1.0


def test_graph_bootstrap_disabled():
    X, y, feats = make_spurious()
    gbm = GradientBoostingRegressor(n_estimators=40, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="noboot")
    out = shortcut_diff(drv, y, edges=[("x", "outcome")], n_perm=0,
                        n_graph_boot=0)
    assert out["graph_bootstrap"] is None
    assert all("class_prob" not in r for r in out["reliance"])


def test_session_and_agent_wiring():
    X, y, feats = make_proxy()
    gbm = GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                    random_state=0).fit(X, y)
    session = MicSession(detect_driver(gbm, feats, X, name="wiring"))
    out = session.shortcut_diff(y, edges=[("x", "outcome"), ("outcome", "p")])
    assert out["verdict"].startswith("SHORTCUT RISK")
    assert session.evidence_log[-1]["tool"] == "shortcut_diff"

    from intervense.agents.interrogator import Interrogator

    session.register_data("y", y.tolist())
    result = Interrogator(session)._execute_tool("shortcut_diff", {})
    assert "verdict" in result and result["verdict"].startswith("SHORTCUT RISK")
