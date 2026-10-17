"""faithfulness verificationlayer:attributionmust"on"model.

 ERASER + random baseline:
  - comprehensiveness comprehensiveness: Top-k(asbackgroundin)prediction
  - sufficiency sufficiency:only Top-k,itspredictionunchanged
  - random baseline: k feature——SHAP notsignificant,not
"""
from __future__ import annotations

import numpy as np

from ..core.types import FaithfulnessReport
from ..drivers.base import MicDriver
from .attribution import explain_sample


def _masked_predict(driver: MicDriver, x: np.ndarray, keep: set[str]) -> float:
    """only keep infeature,itsasbackgroundin."""
    med = np.median(driver.background, axis=0)
    x2 = x.copy()
    for j, f in enumerate(driver.feature_names):
        if f not in keep:
            x2[j] = med[j]
    return driver.predict_one(x2)


def evaluate_faithfulness(
    driver: MicDriver,
    x: np.ndarray,
    attributions=None,
    k: int | None = None,
    n_random: int = 20,
    seed: int = 0,
) -> FaithfulnessReport:
    x = np.asarray(x, dtype=float).ravel()
    if attributions is None:
        attributions = explain_sample(driver, x)
    F = len(driver.feature_names)
    k = k or max(2, int(np.ceil(F * 0.2)))

    p_full = driver.predict_one(x)
    ranked = [a.feature for a in attributions]
    top_k = set(ranked[:k])

    p_comp = _masked_predict(driver, x, keep=set(driver.feature_names) - top_k) # top-k
    p_suff = _masked_predict(driver, x, keep=top_k) # only top-k

    comprehensiveness = abs(p_full - p_comp)
    sufficiency = abs(p_full - p_suff)

    rng = np.random.default_rng(seed)
    random_scores = []
    for _ in range(n_random):
        rand_k = set(rng.choice(ranked, size=k, replace=False).tolist())
        p_rand = _masked_predict(driver, x, keep=set(driver.feature_names) - rand_k)
        random_scores.append(abs(p_full - p_rand))
    random_baseline = float(np.mean(random_scores))

    notes = []
    if comprehensiveness <= random_baseline * 1.2:
        notes.append("warning:SHAP Top-k effectnotrandom baseline,attributionnot")
    if sufficiency > comprehensiveness:
        notes.append("warning:sufficiencycomprehensiveness——Top-k featureinsufficientbythe")
    passed = len(notes) == 0

    return FaithfulnessReport(
        top_k=k,
        comprehensiveness=float(comprehensiveness),
        sufficiency=float(sufficiency),
        random_baseline=random_baseline,
        passed=passed,
        notes=notes,
    )


# ================= sample:model"" =================

def aggregate_faithfulness(
    driver: MicDriver,
    n_samples: int = 20,
    k: int | None = None,
    seed: int = 0,
) -> dict:
    """inbackgrounddistribution n sample,output.

    sampleis;passedandpercentile"thismodelnot".
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(driver.background), size=min(n_samples, len(driver.background)), replace=False)
    comps, suffs, bases, n_pass = [], [], [], 0
    for i in idxs:
        rep = evaluate_faithfulness(driver, driver.sample_at(int(i)), k=k)
        comps.append(rep.comprehensiveness)
        suffs.append(rep.sufficiency)
        bases.append(rep.random_baseline)
        n_pass += int(rep.passed)

    def q(a: list[float], p: float) -> float:
        return float(np.quantile(a, p))

    profile = {
        "n_samples": len(idxs),
        "pass_rate": n_pass / len(idxs),
        "comprehensiveness": {"mean": float(np.mean(comps)), "p25": q(comps, 0.25), "p75": q(comps, 0.75)},
        "sufficiency": {"mean": float(np.mean(suffs)), "p25": q(suffs, 0.25), "p75": q(suffs, 0.75)},
        "random_baseline": {"mean": float(np.mean(bases))},
        "verdict": "" if n_pass / len(idxs) >= 0.8 else "",
    }
    profile["notes"] = (
        []
        if profile["verdict"] == ""
        else [f"passed {profile['pass_rate']:.0%}, 80% threshold——themodelattribution"]
    )
    return profile


# ================= explain -> simulate -> score() =================

def explain_simulate_score(
    driver: MicDriver,
    x: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> dict:
    """OpenAI auto-interp domainmodel.

    explain: model"" = SHAP attribution(feature)
    simulate: intervention experimentmeasureeveryfeaturecausal
               (do(f:=q) backgroundpercentile,predictiondrift)
    score: Spearman gradecorrelation.

    correlation -> attributionmodel;correlation -> .
    and:notdistributioninput,everyallis do intervention.
    """
    x = np.asarray(x, dtype=float).ravel()
    p_base = driver.predict_one(x)
    attrs = explain_sample(driver, x)
    claimed = {a.feature: abs(a.shap_value) for a in attrs}

    measured: dict[str, float] = {}
    for f in driver.feature_names:
        drifts = []
        for qv in quantiles:
            p = driver.predict_one(driver.intervene(x, {f: driver.quantile_of(f, qv)}))
            drifts.append(abs(p - p_base))
        measured[f] = float(np.mean(drifts))

    feats = driver.feature_names
    a = np.array([claimed[f] for f in feats])
    b = np.array([measured[f] for f in feats])
    score = _spearman(a, b)
    return {
        "method": "SHAP vs intervention Spearman correlation",
        "spearman": None if score is None else round(score, 4),
        "verdict": (
            "" if (score is not None and score >= 0.6)
            else "partial" if (score is not None and score >= 0.3)
            else "/verdict"
        ),
        "per_feature": {
            f: {"claimed_importance": round(claimed[f], 4), "measured_sensitivity": round(measured[f], 4)}
            for f in feats
        },
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """ Spearman gradecorrelation(as scipy)."""
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None

    def ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, len(v) + 1)
        return r

    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])
