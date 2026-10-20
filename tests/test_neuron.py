"""GRU interrogationtest:h_T/ + (captum )."""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def gru_session():
    pytest.importorskip("torch")
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_timeseries_demo
    from intervense.drivers.timeseries import TimeSeriesDriver

    net, feats, bg, _ = train_timeseries_demo(epochs=40)
    return MicSession(TimeSeriesDriver(net, feats, bg, name="gru_test"), ConceptPack())


def test_gru_unit_level_mediation(gru_session):
    g = gru_session.activation_graph(0, 1)
    kinds = {e["kind"] for e in g["edges"]}
    assert "layer_mediation" in kinds
    assert "unit_ht_mediation" in kinds # h_T ()
    ht = [e for e in g["edges"] if e["kind"] == "unit_ht_mediation"]
    assert all("@h_T" in e["source"] for e in ht)


def test_gru_persistent_units_valid_semantics(gru_session):
    """ h_T ——."""
    g = gru_session.activation_graph(0, 1)
    ps = [e for e in g["edges"] if e["kind"] == "unit_persistent_mediation"]
    ht_effects = {e["source"].split("#")[1].split("@")[0]: e["effect"]
                  for e in g["edges"] if e["kind"] == "unit_ht_mediation"}
    for e in ps:
        unit = e["source"].split("#")[1].split("@")[0]
        if unit in ht_effects:
            assert abs(e["effect"] - ht_effects[unit]) > 1e-4


def test_neuron_ranking_engine_reported(gru_session):
    nr = gru_session.neuron_ranking(0, top_k=5)
    assert nr["engine"] in ("captum(NeuronGradient)", "fallback(grad×activation)")
    assert nr["n_units"] > 0 and len(nr["top_units"]) == 5
    assert all("unit" in r for r in nr["top_units"])


def test_mlp_activation_graph_unchanged():
    """:MLP unchanged."""
    pytest.importorskip("torch")
    from intervense.demos import FinanceMLP
    from intervense.drivers.torch_model import TorchDriver
    from intervense.analysis.activation_graph import build_activation_graph

    net, feats, bg, _ = FinanceMLP.train(epochs=40)
    drv = TorchDriver(net, feats, bg, name="mlp_guard")
    g = build_activation_graph(drv, 0, 1)
    kinds = {e["kind"] for e in g["edges"]}
    assert "unit_mediation" in kinds and "unit_ht_mediation" not in kinds
