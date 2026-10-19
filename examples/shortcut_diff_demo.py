"""Shortcut diff demo: three synthetic worlds, three audit verdicts.

Scenario 1: only real causes. The audit says SUPPORTED.
Scenario 2: a spurious sibling riding a latent cause plus an overfit
noise feature. The audit says SHORTCUT RISK and classifies both.
Scenario 3: a label echo. The audit classifies it PROXY and reports
the reliance share reading a consequence of the outcome.

Run: python examples/shortcut_diff_demo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from causallens.analysis.shortcut_diff import shortcut_diff
from causallens.drivers import detect_driver


def banner(title: str):
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def report(name: str, out: dict):
    print(f"verdict : {out['verdict']}")
    for r in out["reliance"]:
        print(f"  {r['feature']:8s} share {r['reliance_share']:.1%}  "
              f"{r['classification']}")
    print(f"at risk : {out['reliance_at_risk']:.1%}")


def scenario_1_supported():
    banner("Scenario 1: only real causes")
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.3 * rng.normal(size=n)
    feats = ["a", "b", "filler"]
    model = LinearRegression().fit(X, y)
    drv = detect_driver(model, feats, X, name="s1")
    out = shortcut_diff(drv, y, edges=[("a", "outcome"), ("b", "outcome")])
    report("supported", out)


def scenario_2_spurious():
    banner("Scenario 2: spurious sibling plus overfit noise")
    rng = np.random.default_rng(11)
    n = 800
    c = rng.normal(size=n)
    x = 0.7 * c + rng.normal(scale=0.7, size=n)
    s = 0.7 * c + rng.normal(scale=0.7, size=n)
    y = 0.6 * x + rng.normal(scale=0.6, size=n)
    noise = rng.normal(size=n)
    X = np.column_stack([x, s, noise])
    feats = ["x", "s", "noise"]
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="s2")
    out = shortcut_diff(drv, y, edges=[("x", "outcome")])
    report("spurious", out)


def scenario_3_proxy():
    banner("Scenario 3: label echo, leakage flavored")
    rng = np.random.default_rng(7)
    n = 800
    c = rng.normal(size=n)
    x = 0.8 * c + rng.normal(scale=0.6, size=n)
    s = 0.8 * c + rng.normal(scale=0.6, size=n)
    y = 0.5 * x + rng.normal(scale=0.5, size=n)
    p = y + rng.normal(scale=0.3, size=n)
    noise = rng.normal(size=n)
    X = np.column_stack([x, s, p, noise])
    feats = ["x", "s", "p", "noise"]
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                    random_state=0).fit(X, y)
    drv = detect_driver(gbm, feats, X, name="s3")
    out = shortcut_diff(drv, y, edges=[("x", "outcome"), ("outcome", "p")])
    report("proxy", out)


if __name__ == "__main__":
    scenario_1_supported()
    scenario_2_spurious()
    scenario_3_proxy()
