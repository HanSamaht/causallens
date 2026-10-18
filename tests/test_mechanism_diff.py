"""Crosscoder model diff tests.

Two MLPs train on the same finance data with different seeds and
capacities. The shared dictionary must return shared versus unique
features with per-model causal effects, and leave both patch plans
unwound.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.core import MicSession
from causallens.demos import make_finance_data
from causallens.drivers import detect_driver

torch = pytest.importorskip("torch")


def _train_mlp(hidden, seed, n_rows=1000, epochs=90):
    X, y = make_finance_data(n_rows)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden // 2), torch.nn.ReLU(),
        torch.nn.Linear(hidden // 2, 1), torch.nn.Sigmoid(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        p = net(Xt).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), yt)
        loss.backward()
        opt.step()
    feats = [f"f{i}" for i in range(X.shape[1])]
    return net, feats, X


@pytest.fixture(scope="module")
def two_sessions():
    net_a, feats, X = _train_mlp(20, seed=0)
    net_b, _, _ = _train_mlp(16, seed=5)
    a = MicSession(detect_driver(net_a, feats, X, name="model_a"))
    b = MicSession(detect_driver(net_b, feats, X, name="model_b"))
    return a, b


def test_model_diff_structure(two_sessions):
    a, b = two_sessions
    out = a.mechanism_diff(b, layer="0", epochs=80, n=80, top_k=6)
    assert out["n_tagged"] == out["n_shared"] + out["n_unique_a"] + out["n_unique_b"]
    assert len(out["features"]) > 0
    for f in out["features"]:
        assert f["kind"].startswith(("shared", "unique"))
        assert f["causal_effect_a"] >= 0.0 and f["causal_effect_b"] >= 0.0
        assert len(f["decoder_shares"]) == 2
    assert "model A" in out["verdict"] or "model B" in out["verdict"]


def test_patch_plans_unwound(two_sessions):
    a, b = two_sessions
    a.mechanism_diff(b, layer="0", epochs=40, n=40, top_k=3)
    assert a.driver._patch_plan == {}
    assert b.driver._patch_plan == {}


def test_missing_layer_rejected(two_sessions):
    a, b = two_sessions
    with pytest.raises(ValueError, match="missing in one of the models"):
        a.mechanism_diff(b, layer="not_a_layer", epochs=10, n=10)


def test_layer_autoselect_picks_common(two_sessions):
    a, b = two_sessions
    out = a.mechanism_diff(b, epochs=40, n=40, top_k=3)
    assert out["layer"] in a.driver.list_layers()
    assert out["layer"] in b.driver.list_layers()
