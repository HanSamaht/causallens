"""Tests for bootstrap circuit discovery with confidence intervals."""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def gru_driver():
    pytest.importorskip("torch")
    from intervense.demos import train_timeseries_demo
    from intervense.drivers.timeseries import TimeSeriesDriver
    net, feats, bg, _ = train_timeseries_demo(epochs=60)
    return TimeSeriesDriver(net, feats, bg, name="gru_boot")


def test_bootstrap_circuit_runs(gru_driver):
    """Basic execution."""
    from intervense.analysis.circuit_bootstrap import circuit_with_confidence
    result = circuit_with_confidence(gru_driver, n_bootstrap=5, tolerance=0.02)
    assert "circuit_size_median" in result
    assert "circuit_size_95ci" in result
    assert "units_by_confidence" in result


def test_confidence_bounds(gru_driver):
    """Confidence values should be between 0 and 1."""
    from intervense.analysis.circuit_bootstrap import circuit_with_confidence
    result = circuit_with_confidence(gru_driver, n_bootstrap=5, tolerance=0.02)
    for u in result["units_by_confidence"]:
        assert 0 <= u["confidence"] <= 1


def test_ci_brackets_median(gru_driver):
    """95% CI should contain or bracket the median."""
    from intervense.analysis.circuit_bootstrap import circuit_with_confidence
    result = circuit_with_confidence(gru_driver, n_bootstrap=5, tolerance=0.02)
    lo, hi = result["circuit_size_95ci"]
    median = result["circuit_size_median"]
    assert lo <= hi
    assert lo <= median <= hi
