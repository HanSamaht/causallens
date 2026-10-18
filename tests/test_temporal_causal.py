"""Temporal causal discovery tests: pairwise Granger over channels.

Synthetic ground truth: x1 is an AR(1) driver, x2 follows x1 with one
lag of delay, x3 is independent noise. Granger must link x1 -> x2 and
leave x3 isolated.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.analysis.temporal_causal import (
    granger_pairwise,
    pcmci_edges,
    temporal_causal_graph,
)


def _var_windows(n_series=40, length=200, seed=3, seq_len=20):
    """Windows (n, seq_len, 3) from x1 -> x2 with lag 1, x3 independent."""
    rng = np.random.default_rng(seed)
    windows = []
    for _ in range(n_series):
        x1 = np.zeros(length)
        x2 = np.zeros(length)
        x3 = rng.normal(size=length)
        e1 = rng.normal(0, 0.5, size=length)
        e2 = rng.normal(0, 0.5, size=length)
        for t in range(1, length):
            x1[t] = 0.6 * x1[t - 1] + e1[t]
            x2[t] = 0.9 * x1[t - 1] + e2[t]
        series = np.stack([x1, x2, x3], axis=1)
        for start in range(0, length - seq_len, seq_len):
            windows.append(series[start:start + seq_len])
    return np.asarray(windows, dtype=float)


def test_granger_finds_true_link_and_ignores_noise():
    series = _var_windows()
    results = granger_pairwise(series, max_lag=4, window_stride=1)
    by_pair = {(r["source"], r["target"]): r for r in results}

    x1_to_x2 = by_pair[(0, 1)]
    assert x1_to_x2["p_value"] < 1e-6
    assert x1_to_x2["best_lag"] == 1
    assert x1_to_x2["delta_r2"] > 0.05

    # independent noise must not cause anything at strict significance
    for r in results:
        if 2 in (r["source"], r["target"]):
            assert r["p_value"] > 1e-4, f"noise channel flagged: {r}"


def test_granger_reverse_link_absent():
    series = _var_windows()
    results = granger_pairwise(series, max_lag=4, window_stride=1)
    by_pair = {(r["source"], r["target"]): r for r in results}
    # x2 does not cause x1 at strict significance
    assert by_pair[(1, 0)]["p_value"] > 1e-6


def test_temporal_graph_structure_and_verdict():
    series = _var_windows(n_series=10, seq_len=20)
    driver = _FakeTsDriver(series)
    out = temporal_causal_graph(driver, max_lag=4, alpha=0.05, annotate_model=False)
    assert out["channels"] == ["x1", "x2", "x3"]
    assert out["n_significant"] >= 1
    assert any("x1@lag1 -> x2@now" == e["edge"] for e in out["significant_edges"])
    assert "verdict" in out and "significant lagged edges" in out["verdict"]


def test_temporal_graph_bonferroni_threshold():
    series = _var_windows(n_series=10, seq_len=20)
    driver = _FakeTsDriver(series)
    out = temporal_causal_graph(driver, max_lag=4, alpha=0.05, annotate_model=False)
    n_tests = len(out["edges"])
    for e in out["significant_edges"]:
        assert e["p_corrected"] < 0.05
        assert e["p_value"] * n_tests < 0.05


class _FakeTsDriver:
    """Minimal timeseries driver surface used by temporal_causal_graph."""

    kind = "timeseries"
    seq_len = 20
    n_features_raw = 3
    base_feature_names = ["x1", "x2", "x3"]

    def __init__(self, series):
        self.background = series.reshape(len(series), -1)

    def sample_at(self, i):
        return self.background[i]


def test_negative_control_independent_channels():
    rng = np.random.default_rng(9)
    series = rng.normal(size=(60, 20, 3))
    driver = _FakeTsDriver(series)
    out = temporal_causal_graph(driver, max_lag=4, alpha=0.01, annotate_model=False)
    assert out["n_significant"] <= 1, "independent noise should not yield edges"


def test_end_to_end_with_trained_ts_model():
    pytest.importorskip("torch")
    from causallens.core import MicSession
    from causallens.demos import train_timeseries_demo
    from causallens.drivers.timeseries import TimeSeriesDriver

    model, feats, bg, _ = train_timeseries_demo(epochs=10)
    driver = TimeSeriesDriver(model, feats, bg, name="ts_temporal")
    session = MicSession(driver)

    out = session.temporal_causal_graph(max_lag=4, alpha=0.05, annotate_model=True)
    # demo channels are mutually independent: a clean negative control
    assert out["n_significant"] <= 1
    assert out["channels"] == list(feats)
    # every edge carries the model reliance annotation
    for e in out["edges"]:
        assert "model_reliance" in e
        assert 0.0 <= e["model_reliance"] <= 1.0
    assert "NO TEMPORAL CAUSES" in out["verdict"] or "significant" in out["verdict"]

def _tigramite_available():
    try:
        import tigramite  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tigramite_available(), reason="tigramite is not installed")
def test_pcmci_finds_conditional_link():
    series = _var_windows()
    results = pcmci_edges(series, ["x1", "x2", "x3"], max_lag=4)
    assert results, "expected at least one conditional link"
    top = results[0]
    assert top["p_value"] < 1e-4
    # the planted structure: x1 -> x2 at lag 1 must appear
    assert any(r["source"] == 0 and r["target"] == 1 and r["lag"] == 1
               for r in results if r["p_value"] < 1e-4)
    # the independent noise channel must stay silent
    assert not any(2 in (r["source"], r["target"])
                   for r in results if r["p_value"] < 1e-4)


@pytest.mark.skipif(not _tigramite_available(), reason="tigramite is not installed")
def test_pcmci_through_session():
    pytest.importorskip("torch")
    from causallens.core import MicSession
    from causallens.demos import train_timeseries_demo
    from causallens.drivers.timeseries import TimeSeriesDriver

    model, feats, bg, _ = train_timeseries_demo(epochs=5)
    driver = TimeSeriesDriver(model, feats, bg, name="ts_pcmci")
    session = MicSession(driver)
    out = session.temporal_causal_graph(max_lag=3, method="pcmci",
                                        annotate_model=True, n_windows=64)
    assert out["method"] == "pcmci"
    assert "Runge" in out["reference"]
    # demo channels are independent: a clean negative control
    assert out["n_significant"] <= 1


def test_unknown_method_rejected():
    rng = np.random.default_rng(9)
    series = rng.normal(size=(20, 20, 3))
    driver = _FakeTsDriver(series)
    with pytest.raises(ValueError, match="unknown method"):
        temporal_causal_graph(driver, method="nope", annotate_model=False)
