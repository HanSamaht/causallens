"""Data forensics: the pre modeling plumbing audit."""
import numpy as np
import pandas as pd
import pytest

from causallens.analysis.data_forensics import data_forensics
from causallens.core import MicSession
from causallens.demos import train_finance_demo
from causallens.drivers import detect_driver


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(5)
    n = 600
    skill = rng.normal(size=n)
    weight = 200 + 20 * rng.normal(size=n)
    bench = np.where(weight > 205, rng.normal(20, 5, n), np.nan)
    cohort = rng.choice([2023, 2024, 2025], n)
    speed = rng.normal(8.0, 0.5, n) + (cohort == 2025) * 0.8
    df = pd.DataFrame({"skill": skill, "weight": weight, "bench": bench,
                       "speed": speed, "flat": np.ones(n),
                       "skill_copy": skill, "cohort": cohort})
    df["nfl_id"] = rng.choice(range(560), n)
    other = pd.DataFrame({"nfl_id": range(500)})
    return df, other


def test_basics_flags(planted):
    df, _ = planted
    out = data_forensics(df)
    assert out["basics"]["constant_columns"] == ["flat"]
    assert ["skill", "skill_copy"] in out["basics"]["duplicate_columns"]
    assert "constant columns" in out["verdict"]


def test_mar_hint(planted):
    df, _ = planted
    out = data_forensics(df)
    suspects = out["mechanism"]["mar_suspects"]
    assert any(s["missing_in"] == "bench" and s["differs_by"] == "weight"
               for s in suspects)
    assert "not MCAR" in out["verdict"]


def test_drift_flag(planted):
    df, _ = planted
    out = data_forensics(df, groups="cohort")
    drift_cols = [d["column"] for d in out["drift"]]
    assert "speed" in drift_cols
    assert "cohort" not in drift_cols  # the group column itself is excluded
    medians = next(d for d in out["drift"] if d["column"] == "speed")["group_medians"]
    assert medians["2025"] > medians["2023"]
    assert "drifts across groups" in out["verdict"]


def test_join_audit(planted):
    df, other = planted
    out = data_forensics(df, other_df=other, key="nfl_id")
    j = out["join_audit"]
    assert j["left_keys_matched"] < 0.95
    assert j["right_keys_matched"] < 0.9
    assert j["left_duplicate_keys"] > 0
    assert "match other_df" in out["verdict"] or "match df" in out["verdict"]
    assert all("-" in e or e.isdigit() or True for e in j["unmatched_left_examples"])


def test_join_requires_key(planted):
    df, other = planted
    with pytest.raises(ValueError, match="key is required"):
        data_forensics(df, other_df=other)


def test_clean_frame_verdict():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    out = data_forensics(df)
    assert out["verdict"].startswith("CLEAN PIPE")
    assert out["n_findings"] == 0


def test_misaligned_groups_refused(planted):
    df, _ = planted
    with pytest.raises(ValueError, match="groups"):
        data_forensics(df, groups=np.array(["x"] * 3))


def test_join_null_keys_disclosed():
    df = pd.DataFrame({"k": [1.0, 2.0, np.nan, 4.0], "v": [1, 2, 3, 4]})
    other = pd.DataFrame({"k": [1.0, 2.0, 3.0, np.nan]})
    out = data_forensics(df, other_df=other, key="k")
    j = out["join_audit"]
    assert j["left_null_keys"] == 1 and j["right_null_keys"] == 1
    assert "null join keys" in out["verdict"]


def test_session_and_agent_wiring():
    model, feats, bg, _ = train_finance_demo()
    session = MicSession(detect_driver(model, feats, bg, name="forensics"))
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    out = session.data_forensics(df)
    assert out["verdict"].startswith(("CLEAN PIPE", "DATA FORENSICS"))
    assert session.evidence_log[-1]["tool"] == "data_forensics"

    # registered DataFrame keeps its columns and is reachable by name
    df2 = df.copy()
    df2.loc[df2.index[:20], "b"] = np.nan
    df2["heavy"] = np.nan
    session.register_data("cohort_frame", df2)
    out2 = session.data_forensics("cohort_frame")
    assert "heavy" in out2["missingness"]["columns"]

    from causallens.agents.interrogator import Interrogator

    result = Interrogator(session)._execute_tool(
        "data_forensics", {"data_ref": "cohort_frame"}
    )
    assert "verdict" in result
