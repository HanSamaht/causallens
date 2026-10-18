"""Crosscoder tests: one dictionary across multiple layers.

A two-hidden-layer MLP is trained on the finance demo data. The
Crosscoder must tag features as shared versus layer specific and pass
the simultaneous all-layer causal ablation without touching the model.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.analysis.crosscoder import (
    collect_multilayer_hidden,
    train_crosscoder,
)
from causallens.demos import make_finance_data
from causallens.drivers import detect_driver

torch = pytest.importorskip("torch")


def _train_two_layer_mlp(seed=0, n_rows=1200, epochs=100):
    X, y = make_finance_data(n_rows)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 20), torch.nn.ReLU(),
        torch.nn.Linear(20, 12), torch.nn.ReLU(),
        torch.nn.Linear(12, 1), torch.nn.Sigmoid(),
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
def mlp_driver():
    net, feats, X = _train_two_layer_mlp()
    return detect_driver(net, feats, X, name="mlp_cc")


def test_train_crosscoder_reconstructs_both_sources(mlp_driver):
    hidden = collect_multilayer_hidden(mlp_driver, ["0", "2"], n=80)
    assert hidden["0"].shape == (80, 20)
    assert hidden["2"].shape == (80, 12)
    cc = train_crosscoder(hidden, expand=4, epochs=80)
    for layer in ("0", "2"):
        assert cc["recon_mse"][layer] < 0.1
    assert cc["Z"].shape[0] == 80
    assert 0.0 < cc["mean_active_frac"] <= 1.0


def test_crosscoder_features_tags_and_verifies(mlp_driver):
    from causallens.core import MicSession

    session = MicSession(mlp_driver)
    out = session.crosscoder_features(layers=["0", "2"], epochs=80, n=80, top_k=5)
    assert out["layers"] == ["0", "2"]
    assert out["n_tagged"] == out["n_shared"] + out["n_specific"]
    assert len(out["features"]) > 0
    for f in out["features"]:
        assert f["kind"] in ("shared", "specific")
        assert "decoder_shares" in f and len(f["decoder_shares"]) == 2
        assert isinstance(f["causal_effect"], float)
        assert f["verdict"] in ("passed causal", "correlation only")
    # patch plan must be fully unwound after the run
    assert mlp_driver._patch_plan == {}


def test_single_hidden_layer_model_rejected():
    X, y = make_finance_data(400)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 12), torch.nn.ReLU(),
        torch.nn.Linear(12, 1), torch.nn.Sigmoid(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(30):
        opt.zero_grad()
        p = net(Xt).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), yt)
        loss.backward()
        opt.step()
    driver = detect_driver(net, [f"f{i}" for i in range(X.shape[1])], X, name="mlp1")
    from causallens.core import MicSession

    session = MicSession(driver)
    with pytest.raises(ValueError, match="at least 2 hidden layers"):
        session.crosscoder_features(epochs=10, n=20)


def test_shared_feature_kinds_are_consistent(mlp_driver):
    """decoder_shares must agree with the shared/specific label."""
    hidden = collect_multilayer_hidden(mlp_driver, ["0", "2"], n=60)
    cc = train_crosscoder(hidden, expand=4, epochs=60)
    from causallens.analysis.crosscoder import _feature_source_profile

    checked = 0
    for k in range(cc["n_features"]):
        if (cc["Z"][:, k] > 0.01).sum() < 5:
            continue
        profile = _feature_source_profile(cc["dec_norms"], k)
        weakest = min(profile["decoder_shares"].values())
        if profile["kind"] == "shared":
            assert weakest >= 0.5
        else:
            assert weakest < 0.5
        checked += 1
    assert checked > 0
