"""Expected information gain planner for the next intervention.

The LLM investigator picks tools by prompt discipline alone. This
module adds active experiment design: rank candidate interventions by
how much they are expected to reduce uncertainty about the standing
hypotheses, in the Bayesian optimal experimental design sense.

Mechanics, kept honest and disclosed:
  - candidates are extreme quantile interventions on the top
    attribution features
  - each candidate is previewed on anchor samples with real model
    calls, giving a measured delta
  - the outcome space per candidate is discrete: strong movement or
    not. The measured delta calibrates P(strong), which makes this a
    simulated lookahead ranking, not a blind design
  - a strong outcome moves the log-odds of every hypothesis whose
    statement mentions the intervened feature, by the same bounded
    step the tracker uses, capped at ln(5)
  - EIG = P(strong) times the entropy drop of the hypothetical
    posterior, in nats

Hypothesis to feature matching is literal name containment. That is a
heuristic and the output says so.
"""
from __future__ import annotations

import math

import numpy as np

MATERIALITY = 0.05  # mean prediction shift that counts as strong
MAX_STEP = math.log(5.0)


def _posterior(log_odds: dict[str, float]) -> dict[str, float]:
    m = max(log_odds.values())
    exp = {h: math.exp(v - m) for h, v in log_odds.items()}
    z = sum(exp.values()) or 1.0
    return {h: v / z for h, v in exp.items()}


def _entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 1e-12)


def plan_next_intervention(
    session,
    n_candidates: int = 16,
    n_anchors: int = 8,
    top_k: int = 5,
    seed: int = 0,
) -> dict:
    """Rank candidate interventions by expected information gain.

    Requires hypotheses to be formed first; the planner discriminates
    between hypotheses, without them there is nothing to gain.
    """
    hypotheses = session.investigation.hypotheses
    if not hypotheses:
        raise ValueError(
            "no hypotheses formed; call form_hypotheses before planning"
        )
    driver = session.driver
    rng = np.random.default_rng(seed)

    # candidate features: top mean absolute SHAP over a background slice
    from .attribution import shap_matrix

    bg_slice = driver.background[: min(200, len(driver.background))]
    mean_abs = np.abs(shap_matrix(driver, bg_slice)).mean(axis=0)
    order = np.argsort(-mean_abs)[:n_candidates]
    feats = driver.feature_names

    anchors = rng.choice(
        min(len(driver.background), 200), size=min(n_anchors, len(driver.background)),
        replace=False,
    )

    log_odds_before = {h: hyp.log_odds for h, hyp in hypotheses.items()}
    p_before = _posterior(log_odds_before)
    h_before = _entropy(p_before)

    def mentions(statement: str, feature: str) -> bool:
        return feature in statement

    rows = []
    for j in order:
        f = feats[j]
        lo = driver.quantile_of(f, 0.10)
        hi = driver.quantile_of(f, 0.90)
        deltas = []
        for i in anchors:
            x = driver.sample_at(int(i))
            d = driver.predict_one(driver.intervene(x, {f: hi})) - \
                driver.predict_one(driver.intervene(x, {f: lo}))
            deltas.append(abs(float(d)))
        mean_delta = float(np.mean(deltas))

        touched = [h for h, hyp in hypotheses.items() if mentions(hyp.statement, f)]
        if not touched:
            eig = 0.0
            discriminated = []
        else:
            p_strong = 1.0 / (1.0 + math.exp(-(mean_delta / MATERIALITY - 1.0)))
            step = min(MAX_STEP, MAX_STEP * mean_delta / (MATERIALITY * 4))
            log_odds_after = dict(log_odds_before)
            for h in touched:
                log_odds_after[h] = log_odds_after[h] + step
            p_after = _posterior(log_odds_after)
            eig = p_strong * max(0.0, h_before - _entropy(p_after))
            discriminated = touched

        rows.append({
            "feature": f,
            "intervention": {"from_p10": round(float(lo), 4), "to_p90": round(float(hi), 4)},
            "preview_mean_delta": round(mean_delta, 5),
            "p_strong": round(1.0 / (1.0 + math.exp(-(mean_delta / MATERIALITY - 1.0))), 4)
            if touched else 0.0,
            "eig_nats": round(eig, 5),
            "discriminates": discriminated,
            "call": f"intervene(sample, {{'{f}': {round(float(hi), 4)}}})",
        })

    rows.sort(key=lambda r: -r["eig_nats"])
    top = rows[:top_k]
    best = top[0] if top else None
    if best is None or best["eig_nats"] <= 0:
        verdict = (
            "NO DISCRIMINATING EXPERIMENT AVAILABLE: no candidate feature "
            "appears in any hypothesis statement, or no movement is expected"
        )
    else:
        verdict = (
            f"highest value experiment: intervene on {best['feature']} "
            f"(EIG {best['eig_nats']:.4f} nats), discriminates "
            + (", ".join(best["discriminates"]) if best["discriminates"] else "nothing")
        )

    return {
        "hypotheses": {h: hyp.statement for h, hyp in hypotheses.items()},
        "posterior_before": {h: round(v, 4) for h, v in p_before.items()},
        "entropy_before_nats": round(h_before, 5),
        "candidates": top,
        "verdict": verdict,
        "method": "eig_lookahead_planner",
        "reference": "Foster et al. variational Bayesian optimal experimental design; BED-LLM (2025)",
        "note": (
            "hypothesis to feature matching is literal name containment; "
            "EIG uses a simulated lookahead with the measured delta calibrating "
            "the outcome probability"
        ),
    }
