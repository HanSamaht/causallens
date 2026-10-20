"""Tests for causal mediation decomposition (NDE/NIE)."""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def gru_driver():
    pytest.importorskip("torch")
    from intervense.demos import train_timeseries_demo
    from intervense.drivers.timeseries import TimeSeriesDriver
    net, feats, bg, _ = train_timeseries_demo(epochs=60)
    return TimeSeriesDriver(net, feats, bg, name="gru_med")


def test_mediation_decomposition_runs(gru_driver):
    """Basic execution test."""
    from intervense.analysis.mediation import mediation_decomposition
    result = mediation_decomposition(gru_driver, 0, 1, "gru")
    assert "total_effect" in result
    assert "natural_direct_effect" in result
    assert "natural_indirect_effect" in result


def test_nde_nie_sum(gru_driver):
    """NDE + NIE + interaction should approximately equal TE."""
    from intervense.analysis.mediation import mediation_decomposition
    result = mediation_decomposition(gru_driver, 0, 1, "gru")
    total = result["total_effect"]
    nde = result["natural_direct_effect"]
    nie = result["natural_indirect_effect"]
    interaction = result["interaction"]
    assert abs((nde + nie + interaction) - total) < 1e-5


def test_proportion_mediated_bounds(gru_driver):
    """Proportion mediated should be between 0 and 1."""
    from intervense.analysis.mediation import mediation_decomposition
    result = mediation_decomposition(gru_driver, 0, 1, "gru")
    assert 0 <= result["proportion_mediated"] <= 1


def test_interpretation_present(gru_driver):
    """Should include human-readable interpretation."""
    from intervense.analysis.mediation import mediation_decomposition
    result = mediation_decomposition(gru_driver, 0, 1, "gru")
    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 10
