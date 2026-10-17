"""BacktestDriver test:interrogation.

(2000 bars)test.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causallens.analysis import explain_sample, search_counterfactuals
from causallens.backtest import BTParams, assumption_gap, live_faithfulness, make_synthetic_4h, run_backtest, slippage_stress
from causallens.core import ConceptPack, MicSession
from causallens.drivers.backtest import BacktestDriver


@pytest.fixture(scope="module")
def bt_session():
    ohlc = make_synthetic_4h(2000, seed=7)
    drv = BacktestDriver(ohlc, name="bt_test")
    pack = ConceptPack.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "concepts", "backtest.json"))
    return MicSession(drv, pack)


# ---------- engine:fill semantics ----------

def test_engine_fill_semantics_gap():
    ohlc = make_synthetic_4h(2000, seed=7)
    ladder = run_backtest(ohlc, BTParams(fill_market=0.0))
    market = run_backtest(ohlc, BTParams(fill_market=1.0))
    # :hypothesis PF
    assert ladder["pf"] > market["pf"] * 2
    assert ladder["win_rate"] > market["win_rate"]
    assert market["avg_gap_through"] > 0 #


def test_engine_deterministic():
    ohlc = make_synthetic_4h(500, seed=3)
    a = run_backtest(ohlc, BTParams())
    b = run_backtest(ohlc, BTParams())
    assert a == b


# ---------- driver ----------

def test_backtest_driver_predict(bt_session):
    drv = bt_session.driver
    p = drv.predict(np.array([[0.0, 0.03, 0.15, 2.0, 0.0, 12.0], [1.0, 0.03, 0.15, 2.0, 0.0, 12.0]]))
    assert p[0] > p[1] # PF
    assert drv.prediction_semantics == "score"


def test_backtest_attribution_flags_fill(bt_session):
    attrs = explain_sample(bt_session.driver, bt_session.driver.background[3], bt_session.pack)
    by_f = {a.feature: a for a in attrs}
    assert by_f["fill_market"].role == "spurious" # concept packfill semanticssuspicioushypothesis
    # fill semanticsattribution
    top3 = [a.feature for a in attrs[:3]]
    assert "fill_market" in top3


def test_backtest_counterfactual_threshold_is_one(bt_session):
    # background3(PF<1),counterfactual( 1.0 0)
    drv = bt_session.driver
    paths = search_counterfactuals(drv, drv.background[3])
    assert paths, " PF=1 "
    for p in paths:
        assert (p.cf_prediction - 1.0) * (p.base_prediction - 1.0) < 0


def test_backtest_intervene(bt_session):
    r = bt_session.intervene(3, {"fill_market": 0.0})
    assert r["prediction_after"] > r["prediction_before"] # PF


# ---------- tool ----------

def test_assumption_gap(bt_session):
    g = assumption_gap(bt_session.driver)
    assert g["pf_inflation_ratio"] > 2
    assert "" in g["verdict"]


def test_slippage_stress(bt_session):
    s = slippage_stress(bt_session.driver)
    assert len(s["curve"]) >= 5
    assert s["curve"][0]["slippage_bps"] == 0
    # : PF ()
    pfs = [c["pf"] for c in s["curve"]]
    assert all(b <= a + 1e-9 for a, b in zip(pfs, pfs[1:]))


def test_live_faithfulness_with_real_case():
    # :1m vs = ;backtest vs =
    r1 = live_faithfulness({"pf": 1.09, "win_rate": 0.48}, {"pf": 0.94, "win_rate": 0.45})
    assert r1["verdict"] in {"", ""}
    r2 = live_faithfulness({"pf": 13.2, "win_rate": 0.824}, {"pf": 0.94, "win_rate": 0.45})
    assert r2["verdict"] == ""


# ---------- Session tool ----------

def test_session_backtest_tools(bt_session):
    s = bt_session
    assert s.has_backtest_internals()
    assert "curve" in s.slippage_stress()
    assert "pf_inflation_ratio" in s.assumption_gap()
    lf = s.live_faithfulness({"pf": 1.5}, {"pf": 1.4})
    assert lf["verdict"] == ""
    # evidence
    tools = {e["tool"] for e in s.evidence_log}
    assert {"slippage_stress", "assumption_gap", "live_faithfulness"} <= tools
