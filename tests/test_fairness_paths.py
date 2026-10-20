"""Fairness path decomposition tests.

Synthetic ground truth: protected attribute A moves the label both
directly and through a correlated mediator M. An independent noise
column must not be selected as a mediator.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.analysis.fairness import fairness_path_decomposition
from intervense.drivers import detect_driver


@pytest.fixture(scope="module")
def biased_driver():
    rng = np.random.default_rng(7)
    n = 600
    age = rng.uniform(20, 70, size=n)                    # protected
    income = 3000 + 40 * age + rng.normal(0, 400, size=n) # mediator of age
    noise = rng.normal(size=n)                            # independent
    logit = 0.04 * (income - 4000) + 0.05 * (age - 45)
    y = (logit + rng.normal(0, 0.3, size=n) > 0).astype(int)
    X = np.column_stack([age, income, noise])

    model = LogisticRegression().fit(X, y)
    return detect_driver(model, ["age", "income", "noise"], X, name="fair_paths")


def test_mediator_autodetect_picks_correlated_only(biased_driver):
    out = fairness_path_decomposition(
        biased_driver, list(range(40)), "age", 25.0, 60.0
    )
    assert "income" in out["mediators"]
    assert "noise" not in out["mediators"]
    assert "age" not in out["mediators"]


def test_direct_and_indirect_both_present(biased_driver):
    out = fairness_path_decomposition(
        biased_driver, list(range(40)), "age", 25.0, 60.0
    )
    assert abs(out["paths"]["direct"]["mean_effect"]) > 0.05
    assert abs(out["paths"]["via income"]["mean_effect"]) > 0.05
    # the noise column should carry no path at all
    assert "via noise" not in out["paths"]


def test_decomposition_conserves_total_effect():
    # linear model: no link nonlinearity, so path effects add up exactly
    rng = np.random.default_rng(11)
    n = 500
    age = rng.uniform(20, 70, size=n)
    income = 3000 + 40 * age + rng.normal(0, 400, size=n)
    y = 0.04 * (income - 4000) + 0.05 * (age - 45) + rng.normal(0, 0.3, size=n)
    X = np.column_stack([age, income])
    model = LinearRegression().fit(X, y)
    driver = detect_driver(model, ["age", "income"], X, name="fair_linear")

    out = fairness_path_decomposition(driver, list(range(40)), "age", 25.0, 60.0)
    total = out["mean_total_effect"]
    direct = out["paths"]["direct"]["mean_effect"]
    indirect = out["paths"]["via income"]["mean_effect"]
    assert abs(total - direct - indirect) < 0.01 + 0.02 * abs(total)


def test_verdict_names_the_risk(biased_driver):
    out = fairness_path_decomposition(
        biased_driver, list(range(40)), "age", 25.0, 60.0
    )
    assert out["verdict"].startswith(("DIRECT", "INDIRECT", "NEGLIGIBLE"))


def test_explicit_mediator_list(biased_driver):
    out = fairness_path_decomposition(
        biased_driver, list(range(20)), "age", 25.0, 60.0, mediators=["income"]
    )
    assert out["mediators"] == ["income"]
    assert set(out["paths"]) == {"direct", "via income"}


def test_unknown_feature_rejected(biased_driver):
    with pytest.raises(ValueError, match="unknown feature"):
        fairness_path_decomposition(biased_driver, [0], "height", 1.0, 2.0)
    with pytest.raises(ValueError, match="unknown mediator"):
        fairness_path_decomposition(biased_driver, [0], "age", 1.0, 2.0, mediators=["height"])


def test_session_wiring(biased_driver):
    from intervense.core import MicSession

    session = MicSession(biased_driver)
    out = session.fairness_path_decomposition("age", 25.0, 60.0, n_samples=25)
    assert "verdict" in out and "paths" in out
