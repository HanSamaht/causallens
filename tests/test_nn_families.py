"""Neural family coverage tests: CNN and transformer encoders.

Regression for two real gaps found by probing: user defined nn.Module
subclasses used to fall through driver detection into the sklearn path
and crash, and convolution activation maps broke the Crosscoder shape
handling.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.core import MicSession
from intervense.drivers import detect_driver

torch = pytest.importorskip("torch")


class TinyCNN(torch.nn.Module):
    """Custom subclass: module name is not torch.*, the old detection miss."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(16, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.reshape(-1, 3, 8, 8)
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.conv2(h))
        return self.head(self.pool(h).flatten(1))


def _train(net, X, y, epochs=60, lr=1e-2):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        p = torch.sigmoid(net(Xt)).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), yt)
        loss.backward()
        opt.step()
    return net, float(loss)


def test_custom_nn_module_subclass_detected_as_torch():
    net = TinyCNN()
    X = np.random.default_rng(0).normal(size=(40, 3 * 8 * 8))
    drv = detect_driver(net, [f"px{i}" for i in range(3 * 8 * 8)], X, name="cnn")
    assert drv.kind == "torch"
    assert hasattr(drv, "get_activations")


def test_cnn_full_depth():
    X = np.random.default_rng(1).normal(size=(150, 3 * 8 * 8)).astype(float)
    y = (X.sum(axis=1) > 0).astype(int)
    net, loss = _train(TinyCNN(), X, y, epochs=40)
    assert loss < 0.5  # tool path test, not a model quality test

    session = MicSession(detect_driver(net, [f"px{i}" for i in range(3 * 8 * 8)],
                                       X, name="cnn"))
    r = session.get_attribution(0)
    assert len(r["attributions"]) == 3 * 8 * 8

    session.patch_activation_test(0, 1, "conv2")
    assert session.driver._patch_plan == {}

    out = session.crosscoder_features(layers=["conv1", "conv2"], epochs=30, n=30, top_k=3)
    assert out["n_tagged"] == out["n_shared"] + out["n_specific"]
    for f in out["features"]:
        assert f["causal_effect"] >= 0.0


def test_transformer_encoder():
    class TinyTrans(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Linear(4, 32)
            layer = torch.nn.TransformerEncoderLayer(32, 2, 64, batch_first=True,
                                                     dropout=0.0)
            self.enc = torch.nn.TransformerEncoder(layer, 2)
            self.head = torch.nn.Linear(32, 1)

        def forward(self, x):
            if x.dim() == 2:
                x = x.reshape(-1, 4, 4)
            return self.head(self.enc(self.embed(x)).mean(dim=1))

    X = np.random.default_rng(2).normal(size=(150, 16)).astype(float)
    y = (X[:, :4].sum(axis=1) > 0).astype(int)
    net, loss = _train(TinyTrans(), X, y, epochs=200, lr=3e-3)
    assert loss < 0.3

    session = MicSession(detect_driver(net, [f"f{i}" for i in range(16)], X, name="trans"))
    assert session.driver.kind == "torch"
    r = session.get_attribution(0)
    assert r["attributions"][0]["feature"].startswith("f")

    session.patch_activation_test(0, 1, "enc.layers.0")
    assert session.driver._patch_plan == {}

    out = session.crosscoder_features(layers=["embed", "enc.layers.0"],
                                      epochs=30, n=30, top_k=3)
    assert out["n_tagged"] > 0


def test_conv1d_transcoder_semantics():
    """Conv1d maps (1, C, L) must pool over length, units stay channels.

    Regression for a real bug: a (1, C, L) tensor misread as RNN
    (1, T, H) took the last channel instead, silently producing
    garbage features and zero causal effects.
    """
    from intervense.analysis.crosscoder import flat_last_activation

    a = np.zeros((1, 64, 20), dtype=float)
    a[0, 3, :] = np.arange(1, 21)  # channel 3 has known mean 10.5
    flat = flat_last_activation(a, semantics="conv")
    assert flat.shape == (64,)
    assert flat[3] == pytest.approx(10.5)

    # transformer token-wise semantics: units on the last axis
    b = np.zeros((1, 4, 32), dtype=float)
    b[0, :, 7] = 2.0
    flat_b = flat_last_activation(b, semantics="dense")
    assert flat_b.shape == (32,)
    assert flat_b[7] == pytest.approx(2.0)
