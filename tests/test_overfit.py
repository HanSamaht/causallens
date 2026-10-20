"""DSR + lookaheadtest."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intervense.analysis.lookahead import lookahead_scan
from intervense.analysis.overfit import _norm_cdf, _norm_ppf, deflated_sharpe, sharpe_stats


def test_norm_funs_reciprocal():
    for p in (0.01, 0.05, 0.3, 0.5, 0.7, 0.95, 0.99):
        assert abs(_norm_cdf(_norm_ppf(p)) - p) < 1e-6


def test_dsr_true_edge_survives_noise_edge_dies():
    rng = np.random.default_rng(0)
    good = rng.normal(0.004, 0.01, 500) # edge
    noise = rng.normal(0.0, 0.01, 500) #
    assert deflated_sharpe(good, n_trials=10)["DSR"] > 0.95
    d = deflated_sharpe(noise, n_trials=10)
    assert d["DSR"] < 0.6
    assert "deflated" in d["verdict"] or d["DSR"] < 0.95


def test_sharpe_stats_fields():
    r = np.array([0.01, -0.02, 0.015, 0.005, -0.001, 0.02] * 10)
    st = sharpe_stats(r)
    assert {"n", "sharpe_per_period", "skew", "kurtosis"} <= set(st)


def test_lookahead_scan_finds_injected_lookahead():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.normal(size=n) #
    X_now = rng.normal(size=(n, 4))
    X_prev = X_now.copy()
    X_now[:, 0] = -y + rng.normal(0, 0.05, n) # feature0lookahead(≈)
    X_prev[:, 0] = rng.normal(size=n) #
    r = lookahead_scan(y, X_now, X_prev, [f"f{i}" for i in range(4)])
    assert r["leak_suspects"] and r["leak_suspects"][0]["feature"] == "f0"
    assert r["leak_suspects"][0]["survival_ratio"] < 0.35

    X_clean = rng.normal(size=(n, 4)) # feature:correlation()
    X_prev_clean = X_clean + rng.normal(0, 0.1, (n, 4))
    r2 = lookahead_scan(y, X_clean, X_prev_clean, [f"f{i}" for i in range(4)])
    assert not r2["leak_suspects"]
