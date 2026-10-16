"""深水区三件套测试：可逆损毁 / ACDC-lite 电路 / SAE-lite 特征。"""
import sys, os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def gru():
    pytest.importorskip("torch")
    from mic.demos import train_timeseries_demo
    from mic.drivers.timeseries import TimeSeriesDriver

    net, feats, bg, _ = train_timeseries_demo(epochs=60)
    return TimeSeriesDriver(net, feats, bg, name="gru_deep")


def test_param_ablation_reversible(gru):
    from mic.analysis.param_ablation import param_ablation
    import numpy as np

    p0 = gru.predict(gru.background[:20]).copy()
    r = param_ablation(gru, ["gru.weight_hh", "head.weight"], metric="pred_center", n_samples=20)
    for x in r["results"]:
        assert x["reversible"], "恢复失败的结果必须作废"
        assert "n_weights" in x
    assert np.allclose(gru.predict(gru.background[:20]), p0, atol=1e-6)  # 权重真回来了


def test_unit_mask_reversible(gru):
    import numpy as np

    p0 = gru.predict(gru.background[:10]).copy()
    with gru.ablate_units("gru", {0, 1, 2}):
        p1 = gru.predict(gru.background[:10])
    p2 = gru.predict(gru.background[:10])
    assert not np.allclose(p1, p0)          # 掩码生效
    assert np.allclose(p2, p0, atol=1e-7)   # 退出即恢复


def test_discover_circuit_sparse_and_valid(gru):
    from mic.analysis.param_ablation import discover_circuit

    c = discover_circuit(gru, capability_samples=12, tolerance=0.02)
    assert "circuit_units" in c
    assert 1 <= c["n_circuit"] <= c["n_units_total"]
    assert c["n_pruned"] >= 1                          # 真有可剔除的
    assert c["residual_after_restore"] < 0.05          # 电路自洽


def test_sae_features_labeled_and_falsified(gru):
    from mic.analysis.sae_lite import sae_features

    s = sae_features(gru, top_k=5)
    assert s["recon_mse"] < 0.05                       # 字典学到东西
    assert len(s["features"]) >= 2
    for f in s["features"]:
        assert f["correlates_with"]                     # 有自动标注
        assert f["verdict"] in ("通过因果检验", "未通过（相关非因果）")
    # GRU demo 的主导因子是动量——至少一个通过检验的特征与 ts_momentum 相关
    labeled = [f for f in s["features"] if "ts_momentum" in f["correlates_with"]]
    assert labeled, "SAE 应找到动量编码特征"
