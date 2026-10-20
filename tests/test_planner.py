"""EIG planner tests: the next experiment should target the real driver.

Hypotheses name different features. The candidate touching the actual
strong driver must outrank the candidate touching a weak feature,
because its expected outcome separates the hypotheses harder.
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


@pytest.fixture(scope="module")
def session():
    model, feats, bg, _ = train_finance_demo()
    return MicSession(detect_driver(model, feats, bg, name="fin_plan"))


def test_requires_hypotheses(session):
    from intervense.analysis.experiment_planner import plan_next_intervention

    with pytest.raises(ValueError, match="form_hypotheses"):
        plan_next_intervention(session)


def test_planner_ranks_real_driver_first(session):
    session.form_hypotheses([
        "momentum drives the lending style decision",
        "month_index seasonality drives the decision",
    ])
    from intervense.analysis.experiment_planner import plan_next_intervention

    out = plan_next_intervention(session, n_candidates=8, top_k=5, seed=0)
    assert "candidates" in out and out["candidates"]
    best = out["candidates"][0]
    # momentum is the planted strong feature in the demo generator
    assert best["feature"] == "momentum"
    assert best["eig_nats"] > 0
    assert "H1" in best["discriminates"]
    weaker = [c for c in out["candidates"] if c["feature"] == "month_index"]
    if weaker:
        assert best["eig_nats"] > weaker[0]["eig_nats"]
    assert out["verdict"].startswith("highest value experiment")


def test_eig_bounded_and_deterministic(session):
    session.investigation.hypotheses = {}
    session.form_hypotheses([
        "momentum drives the decision",
        "size drives the decision",
    ])
    from intervense.analysis.experiment_planner import plan_next_intervention

    a = plan_next_intervention(session, n_candidates=6, top_k=3, seed=1)
    b = plan_next_intervention(session, n_candidates=6, top_k=3, seed=1)
    assert [c["feature"] for c in a["candidates"]] == [c["feature"] for c in b["candidates"]]
    for c in a["candidates"]:
        assert 0.0 <= c["eig_nats"] <= 1.39  # ln 4, max entropy over 2 hypotheses


def test_interrogator_exposes_planner(session):
    from intervense.agents.interrogator import Interrogator

    inter = Interrogator(session)
    names = {t["function"]["name"] for t in inter.tool_schemas()}
    assert "plan_next_intervention" in names
