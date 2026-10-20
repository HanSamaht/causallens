"""leakagecontaminationtest."""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intervense.analysis.leakage import duplicate_overlap, label_distribution_check, single_feature_leak_scan


def test_duplicate_overlap_clean_and_contaminated():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(300, 6))
    B_clean = rng.normal(size=(100, 6)) + 5.0
    assert duplicate_overlap(A, B_clean)["verdict"].startswith("")

    B_bad = np.vstack([A[:20], rng.normal(size=(80, 6)) * 0.01]) # 20% contamination
    r = duplicate_overlap(A, B_bad)
    assert r["exact_duplicates"] == 20
    assert r["verdict"] != ""


def test_leak_scan_finds_injected_leak():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(800, 5))
    y = (rng.random(800) > 0.5).astype(int)
    X[:, 0] = y + rng.normal(0, 0.05, 800) # leakagefeature:=
    r = single_feature_leak_scan(y, X, [f"f{i}" for i in range(5)])
    assert r["leak_suspects"] and r["leak_suspects"][0]["feature"] == "f0"
    assert r["leak_suspects"][0]["solo_auc"] > 0.9

    X2 = rng.normal(size=(800, 5)) # leakage
    r2 = single_feature_leak_scan(y, X2, [f"f{i}" for i in range(5)])
    assert not r2["leak_suspects"]


def test_label_check():
    a = np.array([1] * 50 + [0] * 50)
    b = np.array([1] * 49 + [0] * 51)
    assert label_distribution_check(a, b)["verdict"] == "consistent"
    c = np.array([1] * 80 + [0] * 20)
    assert "drift" in label_distribution_check(a, c)["verdict"]


def test_session_tools_wired():
    from intervense.core import ConceptPack, MicSession
    from intervense.demos import train_finance_demo
    from intervense.drivers import detect_driver

    model, feats, bg, _ = train_finance_demo()
    s = MicSession(detect_driver(model, feats, bg), ConceptPack())
    r = s.check_contamination(new_data=bg[:50] + 10.0) # background ->
    assert "overlap" in r
