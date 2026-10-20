"""Causal prior injection tests.

A prior with known ground truth on the finance demo: momentum drives
the model strongly (the demo is built that way) and size does not.
The prior must be nudged, logged, bounded, and testable.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.core import MicSession
from intervense.demos import FINANCE_FEATURES, train_finance_demo
from intervense.drivers import detect_driver

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def session():
    model, feats, bg, _ = train_finance_demo()
    driver = detect_driver(model, feats, bg, name="fin_prior")
    return MicSession(driver)


def test_prior_set_and_tested_against_interventions(session):
    out = session.set_causal_prior([
        {"source": "momentum", "target": "prediction", "strength": "strong"},
        {"source": "size", "target": "prediction", "strength": "weak",
         "expected": False},
    ])
    assert out["n_edges"] == 2

    report = session.test_causal_prior()
    verdicts = {r["edge"]: r["verdict"] for r in report["rows"]}
    assert verdicts["momentum -> prediction"] == "supported"
    assert verdicts["size -> prediction"] in ("clean", "supported")
    assert "PRIOR" in report["verdict"]


def test_prior_nudges_hypotheses_bounded_and_logged(session):
    fresh = session.investigation
    fresh.hypotheses = {}
    fresh.set_prior([
        {"source": "momentum", "target": "prediction", "strength": "strong"},
    ])
    snap = fresh.form([
        "momentum is the main driver of the prediction",
        "month index seasonality drives the prediction",
    ])
    h1 = snap["hypotheses"]["H1"]
    h2 = snap["hypotheses"]["H2"]
    assert h1["posterior"] > h2["posterior"]
    # bounded: the nudge never exceeds ln(1.5)
    assert h1["evidence"][0]["tool"] == "prior_injection"
    assert h1["evidence"][0]["lr_step"] <= round(math.log(1.5), 3)


def test_bad_prior_edges_rejected(session):
    with pytest.raises(ValueError, match="unknown prior"):
        session.set_causal_prior([{"source": "height", "target": "prediction"}])
    with pytest.raises(ValueError, match="strength"):
        session.set_causal_prior([
            {"source": "momentum", "target": "prediction", "strength": "huge"}
        ])


def test_test_prior_without_prior_rejected(session):
    saved = session.investigation.causal_prior
    session.investigation.causal_prior = []
    try:
        with pytest.raises(ValueError, match="no prior stored"):
            session.test_causal_prior()
    finally:
        session.investigation.causal_prior = saved


def test_feature_to_feature_edge_verdicts(session):
    session.investigation.set_prior([
        {"source": "momentum", "target": "size", "strength": "weak"},
    ])
    report = session.test_causal_prior()
    v = report["rows"][0]["verdict"]
    assert v in ("supported", "reversed", "untested")


def test_interrogator_exposes_prior_tools(session):
    from intervense.agents.interrogator import Interrogator

    inter = Interrogator(session)
    names = {t["function"]["name"] for t in inter.tool_schemas()}
    assert "set_causal_prior" in names
    assert "test_causal_prior" in names
