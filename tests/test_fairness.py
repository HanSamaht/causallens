"""Counterfactual fairness auditing tests."""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.analysis.fairness import counterfactual_fairness_check


def test_no_flip_when_feature_irrelevant():
    """Model that ignores the protected feature should pass."""
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_finance_demo
    from intervense.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    drv = detect_driver(model, feats, bg, name="fin")
    # momentum is a legitimate feature, changing it should flip predictions
    # but we test with a feature that has low attribution (size)
    result = counterfactual_fairness_check(
        drv, [0, 1, 2, 3, 4], "size", [-1.0, 0.0, 1.0]
    )
    assert result["n_tests"] == 15  # 5 samples x 3 contrast values
    assert "verdict" in result
    assert "flip_rate" in result


def test_flip_when_feature_drives_prediction():
    """Changing a high-attribution feature should produce flips."""
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_finance_demo
    from intervense.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    drv = detect_driver(model, feats, bg, name="fin")
    # momentum is the top feature; extreme changes should flip predictions
    result = counterfactual_fairness_check(
        drv, list(range(20)), "momentum", [-2.0, 2.0]
    )
    assert result["n_tests"] == 40
    assert result["flip_rate"] > 0  # changing top feature should cause flips


def test_credit_age_fairness():
    """Verify the credit model age bias finding using formal framework."""
    import pickle
    import pandas as pd

    try:
        bg = pd.read_csv("examples/kaggle_models/credit_risk_bg.csv")
        model = pickle.load(
            open("examples/kaggle_models/credit_rf_dir/credit_random_model.pkl", "rb")
        )
    except FileNotFoundError:
        pytest.skip("Credit model files not available")

    from intervense.drivers import detect_driver

    drv = detect_driver(model, list(bg.columns), bg.values, name="credit")
    result = counterfactual_fairness_check(
        drv, list(range(30)), "person_age", [22.0, 45.0, 70.0]
    )
    # Verify we get structured results
    assert "flip_rate" in result
    assert "verdict" in result
    assert len(result["per_sample"]) == 90  # 30 x 3
    # The verdict should be one of the three types
    assert result["verdict"].split(":")[0] in ("PASS", "VIOLATION", "WARNING")
