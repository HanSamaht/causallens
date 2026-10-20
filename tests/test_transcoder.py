"""Transcoder tests: block dictionaries with panel data.

A two-hidden-layer MLP over the finance demo data. The transcoder
must learn input to output rules for the first block, verify features
causally, and carry activation examples with valid sample indices.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.demos import make_finance_data

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def mlp_driver():
    from intervense.drivers import detect_driver

    X, y = make_finance_data(1200)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 20), torch.nn.ReLU(),
        torch.nn.Linear(20, 12), torch.nn.ReLU(),
        torch.nn.Linear(12, 1), torch.nn.Sigmoid(),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(90):
        opt.zero_grad()
        p = net(Xt).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), yt)
        loss.backward()
        opt.step()
    return detect_driver(net, [f"f{i}" for i in range(X.shape[1])], X, name="mlp_tc")


def test_transcoder_trains_and_tags(mlp_driver):
    from intervense.analysis.transcoder import transcoder_features

    out = transcoder_features(mlp_driver, block=("0", "2"), epochs=80, n=80, top_k=5)
    assert out["block"] == {"input": "0", "output": "2"}
    assert out["recon_mse"] < 0.2
    assert out["n_tagged"] > 0
    for f in out["features"]:
        assert isinstance(f["causal_effect"], float)
        assert f["verdict"] in ("passed causal", "correlation only")
        examples = f["activation_examples"]
        assert 0 < len(examples) <= 5
        for ex in examples:
            assert 0 <= ex["sample_index"] < 80
            assert ex["activation"] > 0


def test_transcoder_unwinds_patch_plan(mlp_driver):
    from intervense.core import MicSession

    session = MicSession(mlp_driver)
    session.transcoder_features(block=("0", "2"), epochs=30, n=40, top_k=3)
    assert mlp_driver._patch_plan == {}


def test_transcoder_single_module_rejected():
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU())
    from intervense.drivers import detect_driver

    driver = detect_driver(net, ["a", "b", "c", "d"],
                           np.random.default_rng(0).normal(size=(30, 4)), name="one")
    from intervense.analysis.transcoder import transcoder_features

    with pytest.raises(ValueError, match="block"):
        transcoder_features(driver, epochs=5, n=10)


def test_crosscoder_rows_carry_examples(mlp_driver):
    from intervense.analysis.crosscoder import crosscoder_features

    out = crosscoder_features(mlp_driver, layers=["0", "2"], epochs=40, n=60, top_k=3)
    for f in out["features"]:
        assert "activation_examples" in f
        assert all(ex["activation"] > 0 for ex in f["activation_examples"])
