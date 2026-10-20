"""Diverse counterfactual tests: DiCE-style multi-path search.

Each returned path must flip the prediction and use a distinct
feature subset from the other paths.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.core import MicSession
from intervense.demos import train_finance_demo
from intervense.drivers import detect_driver

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def finance_env():
    model, feats, bg, auc = train_finance_demo()
    driver = detect_driver(model, feats, bg, name="fin_diverse")
    session = MicSession(driver)
    return session


def _first_sample_with_paths(session, n_paths, scan=30):
    for i in range(scan):
        paths = session.diverse_counterfactuals(i, n_paths=n_paths)
        if paths:
            return i, paths
    return 0, []


def test_diverse_paths_flip_and_are_distinct(finance_env):
    session = finance_env
    _, paths = _first_sample_with_paths(session, n_paths=3)
    assert paths, "expected at least one counterfactual path in 30 samples"

    feature_sets = []
    for p in paths:
        assert p["flipped"] is True
        assert p["prediction_after"] in (0.0, 1.0) or 0.0 <= p["prediction_after"] <= 1.0
        assert isinstance(p["changed_features"], dict) and p["changed_features"]
        feature_sets.append(frozenset(p["changed_features"].keys()))

    # Diversity contract: no two paths change the exact same feature set
    assert len(feature_sets) == len(set(feature_sets))


def test_diverse_paths_actually_flip_when_applied(finance_env):
    session = finance_env
    i, paths = _first_sample_with_paths(session, n_paths=3)
    if not paths:
        pytest.skip("no flips available in scanned samples")
    driver = session.driver
    x = driver.sample_at(i)
    p_base = driver.predict_one(x)
    base_label = 1 if p_base > 0.5 else 0
    for p in paths:
        x2 = x.copy()
        for f, v in p["changed_features"].items():
            x2[driver.feature_names.index(f)] = v
        p_new = driver.predict_one(x2)
        new_label = 1 if p_new > 0.5 else 0
        assert new_label != base_label


def test_diverse_no_paths_for_unflippable_sample(finance_env):
    session = finance_env
    # A confident sample deep in one class should return a list (possibly empty)
    paths = session.diverse_counterfactuals(0, n_paths=2)
    assert isinstance(paths, list)
    for p in paths:
        assert p["flipped"] is True
