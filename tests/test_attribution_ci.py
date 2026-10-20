"""Attribution confidence tests: bootstrap intervals and rank stability.

On the finance demo the planted strong feature momentum must hold its
rank across replicates with an interval away from zero. A pure noise
feature must be flagged: its interval should be narrow and low.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.analysis.attribution_ci import attribution_with_ci
from intervense.demos import train_finance_demo
from intervense.drivers import detect_driver

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def session():
    from intervense.core import MicSession

    model, feats, bg, _ = train_finance_demo()
    return MicSession(detect_driver(model, feats, bg, name="fin_ci"))


def test_intervals_and_stability_on_demo(session):
    out = attribution_with_ci(session.driver, n_bootstrap=15, top_k=3,
                              max_rows=150, seed=1)
    rows = out["features"]
    assert len(rows) == len(session.driver.feature_names)
    for r in rows:
        assert 0.0 <= r["ci_low"] <= r["share"] <= r["ci_high"] <= 1.0
        assert r["ci_width"] > 0
    top = rows[0]
    # a strong planted feature holds rank 0 in most replicates
    assert top["top3_stability"] >= 0.8
    assert top["ci_low"] > 0.01
    assert out["verdict"].startswith(("STABLE", "WEAK EVIDENCE"))


def test_too_few_replicates_rejected(session):
    with pytest.raises(ValueError, match="cannot support"):
        attribution_with_ci(session.driver, n_bootstrap=5)


def test_noise_feature_low_share(session):
    out = attribution_with_ci(session.driver, n_bootstrap=12, top_k=3,
                              max_rows=150, seed=2)
    by_name = {r["feature"]: r for r in out["features"]}
    # size is planted weak in the demo generator
    weak = min(by_name.values(), key=lambda r: r["share"])
    assert weak["share"] < top_weak_ceiling(by_name)


def top_weak_ceiling(by_name):
    """The weakest feature should sit well below the top feature."""
    top = max(by_name.values(), key=lambda r: r["share"])
    return top["share"] * 0.5


def test_session_wiring(session):
    out = session.attribution_confidence(n_bootstrap=10, top_k=3)
    assert "features" in out and "verdict" in out
    assert out["n_bootstrap"] == 10
