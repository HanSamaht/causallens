"""Identifiability tests: backdoor and front-door decisions via DoWhy.

Ground truth data: z confounds x and the label, true direct effect of
x is 0.5. The auto framework graph must find the backdoor set {z}; an
explicit user graph is honored; a latent common cause between treatment
and outcome is refused instead of estimated naively.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.analysis.identifiability import HAS_DOWHY, identify_real_effect
from causallens.drivers import detect_driver

pytestmark = pytest.mark.skipif(not HAS_DOWHY, reason="DoWhy is not installed")


@pytest.fixture(scope="module")
def confounded_driver():
    rng = np.random.default_rng(0)
    n = 800
    z = rng.normal(size=n)
    x = 0.8 * z + rng.normal(0, 0.5, size=n)
    y = 0.5 * x + 0.7 * z + rng.normal(0, 0.2, size=n)
    m = LinearRegression().fit(np.column_stack([x, z]), y)
    return detect_driver(m, ["x", "z"], np.column_stack([x, z]), name="ident")


def test_auto_graph_finds_backdoor(confounded_driver):
    out = identify_real_effect(confounded_driver, "x")
    assert out["identified"] is True
    assert out["backdoor_set"] == ["z"]
    assert out["estimate"] is not None
    # estimate is the p25 to p75 contrast: 0.5 slope times about 1.4 spread
    assert 0.4 < out["estimate"] < 1.0


def test_explicit_graph_honored(confounded_driver):
    out = identify_real_effect(
        confounded_driver, "x",
        edges=[("z", "x"), ("z", "prediction"), ("x", "prediction")],
    )
    assert out["graph_provenance"].startswith("user supplied")
    assert out["backdoor_set"] == ["z"]
    assert out["identified"] is True


def test_latent_common_cause_refused(confounded_driver):
    # outcome z correlates with treatment x in the data; the auto graph
    # marks that correlation as a latent common cause, and the effect of
    # x on z must be refused rather than estimated naively
    out = identify_real_effect(confounded_driver, "x", outcome="z")
    assert out["identified"] is False
    assert out["estimate"] is None
    assert "NOT IDENTIFIED" in out["verdict"]


def test_unknown_names_rejected(confounded_driver):
    with pytest.raises(ValueError, match="unknown treatment"):
        identify_real_effect(confounded_driver, "height")
    with pytest.raises(ValueError, match="unknown outcome"):
        identify_real_effect(confounded_driver, "x", outcome="weather")


def test_session_wiring(confounded_driver):
    from causallens.core import MicSession

    session = MicSession(confounded_driver)
    out = session.identify_real_effect("x")
    assert "identified" in out and "verdict" in out
