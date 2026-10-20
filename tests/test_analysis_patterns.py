"""Analysis pattern absorption: panel FE, mediation chain, holdout, assertions."""
import numpy as np
import pandas as pd
import pytest

from intervense.analysis.data_forensics import data_forensics
from intervense.analysis.holdout_evaluation import holdout_evaluation
from intervense.analysis.mediation_chain import mediation_chain
from intervense.analysis.panel_fixed_effects import panel_fixed_effects
from intervense.core import MicSession
from intervense.demos import train_finance_demo
from intervense.drivers import detect_driver


def make_panel(n_units=60, reps=30, within=0.6, between=2.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        unit_effect = rng.normal(0, between)
        for _ in range(reps):
            x = rng.normal()
            noise = rng.normal(0, 0.5)
            rows.append({"unit": f"u{u}", "x": x,
                         "y": within * x + unit_effect + noise})
    return pd.DataFrame(rows)


def test_panel_fe_recovers_within_effect():
    d = make_panel()
    out = panel_fixed_effects(d, "y", ["x"], group="unit")
    assert "WITHIN-EFFECTS" in out["verdict"]
    row = out["features"][0]
    # standardized coefficient: 0.6 * sd(x)/sd(y) territory, positive
    assert row["coef_within_per_sd"] > 0.15
    assert row["p_value"] < 0.01
    # clustering penalty visible: cluster SE exceeds naive SE
    assert row["se_cluster"] >= 0.9 * row["se_naive"]


def test_panel_fe_absorbs_between_confounding():
    rng = np.random.default_rng(2)
    unit_effects = {f"u{u}": rng.normal(0, 2.0) for u in range(60)}
    d = make_panel(seed=1)
    d["x"] = rng.normal(size=len(d))
    d["y"] = d["unit"].map(unit_effects) + rng.normal(0, 0.5, size=len(d))
    out = panel_fixed_effects(d, "y", ["x"], group="unit")
    assert "NO WITHIN-EFFECTS" in out["verdict"]


def test_panel_fe_few_clusters_warns():
    rng = np.random.default_rng(3)
    rows = []
    for u in range(5):
        unit_effect = rng.normal(0, 2.0)
        for _ in range(40):
            x = rng.normal()
            rows.append({"unit": f"u{u}", "x": x,
                         "y": 1.5 * x + unit_effect + rng.normal(0, 0.4)})
    d = pd.DataFrame(rows)
    out = panel_fixed_effects(d, "y", ["x"], group="unit")
    assert "FRAGILE" in out["verdict"]


def make_mediation(a=0.5, b=0.8, direct=0.0, n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.normal(size=n)
    m = a * t + rng.normal(size=n)
    y = direct * t + b * m + rng.normal(size=n)
    return pd.DataFrame({"t": t, "m": m, "y": y})


def test_mediation_full():
    d = make_mediation(direct=0.0)
    out = mediation_chain(d, "t", "m", "y", n_boot=500, seed=0)
    assert out["verdict"].startswith("MEDIATION ESTABLISHED")
    lo, hi = out["indirect_ci95"]
    assert lo > 0
    assert out["mediated_share"] > 0.85  # direct is zero: full mediation


def test_mediation_partial():
    d = make_mediation(direct=0.6)
    out = mediation_chain(d, "t", "m", "y", n_boot=500, seed=1)
    assert out["verdict"].startswith("MEDIATION ESTABLISHED")
    assert 0.3 < out["mediated_share"] < 0.8


def test_mediation_suppression_guard():
    # weak, opposite-signed total with a strong indirect path: the
    # share ratio must be refused, not reported as 123 percent
    rng = np.random.default_rng(4)
    n = 400
    t = rng.normal(size=n)
    m = 0.9 * t + rng.normal(size=n)
    y = -0.25 * t + 0.9 * m + rng.normal(size=n)
    d = pd.DataFrame({"t": t, "m": m, "y": y})
    out = mediation_chain(d, "t", "m", "y", n_boot=500, seed=0)
    assert out["verdict"].startswith("SUPPRESSION")


def test_mediation_null():
    rng = np.random.default_rng(5)
    n = 400
    t, m = rng.normal(size=n), rng.normal(size=n)
    y = rng.normal(size=n)
    out = mediation_chain(pd.DataFrame({"t": t, "m": m, "y": y}),
                          "t", "m", "y", n_boot=1000, seed=0)
    assert out["verdict"].startswith(("NOT ESTABLISHED", "SUPPRESSION"))


def test_holdout_evaluation():
    rng = np.random.default_rng(6)
    rows = []
    for season in (2023, 2024, 2025):
        for _ in range(400):
            x1, x2 = rng.normal(size=2)
            p = 1 / (1 + np.exp(-(1.2 * x1 + 0.6 * x2)))
            rows.append({"season": season, "x1": x1, "x2": x2,
                         "y": int(rng.random() < p)})
    d = pd.DataFrame(rows)
    out = holdout_evaluation(d, "y", ["x1", "x2"], group="season",
                             holdout_groups=[2025])
    assert "STRONG HOLDOUT" in out["verdict"]
    assert out["auc"] > 0.7
    assert out["brier"] < out["brier_prior_baseline"]
    assert len(out["calibration"]) >= 4


def test_plausibility_assertion_catches_phantom_distance():
    # the exact lesson: distance-to-QB on pressure plays must be
    # small; a features bug produced a median of 9.4
    rng = np.random.default_rng(7)
    n = 300
    d = pd.DataFrame({
        "pressure_event": rng.integers(0, 2, n),
        "min_dist": rng.uniform(0, 18, n)})
    out = data_forensics(d, assertions=[{
        "name": "pressure plays close to the target",
        "column": "min_dist", "statistic": "median", "op": "<",
        "value": 2.0, "where": ["pressure_event", 1]}])
    assert out["plausibility"]["violations"]
    assert "PLAUSIBILITY VIOLATION" in out["verdict"]


def test_plausibility_assertion_passes_clean():
    rng = np.random.default_rng(8)
    n = 300
    press = rng.integers(0, 2, n)
    d = pd.DataFrame({
        "pressure_event": press,
        "min_dist": np.where(press == 1, rng.uniform(0, 1.5, n),
                             rng.uniform(0, 15, n))})
    out = data_forensics(d, assertions=[{
        "name": "pressure plays close to the target",
        "column": "min_dist", "statistic": "median", "op": "<",
        "value": 2.0, "where": ["pressure_event", 1]}])
    assert out["plausibility"]["violations"] == []


def test_session_wiring_for_new_tools():
    model, feats, bg, _ = train_finance_demo()
    session = MicSession(detect_driver(model, feats, bg, name="wire2"))
    d = make_mediation(direct=0.3, n=500)
    session.register_data("med_df", d)
    out = session.mediation_chain("med_df", "t", "m", "y", n_boot=200)
    assert out["verdict"].startswith(("MEDIATION", "NOT", "SUPPRESSION"))
    assert session.evidence_log[-1]["tool"] == "mediation_chain"

    from intervense.agents.interrogator import Interrogator

    r = Interrogator(session)._execute_tool(
        "mediation_chain",
        {"data_ref": "med_df", "treatment": "t", "mediator": "m",
         "outcome": "y"})
    assert "verdict" in r
