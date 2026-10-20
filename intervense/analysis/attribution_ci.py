"""Bootstrap confidence intervals for attribution.

SHAP values are point estimates computed on one background sample.
Sampling noise moves them, and two features whose shares differ by a
decimal can swap ranks on a different draw. This module resamples the
background rows, recomputes the attribution, and reports percentile
intervals plus rank stability: the fraction of replicates in which a
feature keeps its top-k position.

The point is honesty about strength of evidence: a feature whose share
interval touches zero is not a finding, and a rank that flips across
replicates is not a ranking.
"""
from __future__ import annotations

import numpy as np

from .attribution import shap_matrix


def attribution_with_ci(
    driver,
    n_bootstrap: int = 50,
    sample_frac: float = 0.8,
    top_k: int = 5,
    seed: int = 0,
    max_rows: int = 300,
) -> dict:
    """Bootstrap the mean absolute SHAP share per feature.

    Args:
        driver: any Intervense driver.
        n_bootstrap: number of resamples. 50 is the practical floor for
            a 95 percent percentile interval.
        sample_frac: rows drawn per replicate, without replacement.
        top_k: rank stability window, fraction of replicates the
            feature stays within the top k.
        seed: reproducibility.
        max_rows: cap on rows used per replicate, runtime guard.

    Returns:
        Dict with per feature intervals, shares, rank stability, and a
        verdict naming which top features are not separated from zero
        or from each other.
    """
    if n_bootstrap < 10:
        raise ValueError("n_bootstrap below 10 cannot support a percentile interval")
    bg = driver.background
    n_rows = min(len(bg), max_rows)
    m = max(int(n_rows * sample_frac), 10)
    rng = np.random.default_rng(seed)
    feats = list(driver.feature_names)

    shares = np.zeros((n_bootstrap, len(feats)))
    for b in range(n_bootstrap):
        idx = rng.choice(n_rows, size=m, replace=False)
        sm = shap_matrix(driver, bg[idx])
        mean_abs = np.abs(sm).mean(axis=0)
        shares[b] = mean_abs / (mean_abs.sum() + 1e-12)

    lo = np.percentile(shares, 2.5, axis=0)
    hi = np.percentile(shares, 97.5, axis=0)
    point = shares.mean(axis=0)

    rank_matrix = np.argsort(-shares, axis=1)
    stability = np.zeros(len(feats))
    for j in range(len(feats)):
        # rank of feature j in each replicate
        ranks = np.array([np.where(rank_matrix[b] == j)[0][0] for b in range(n_bootstrap)])
        stability[j] = float((ranks < top_k).mean())

    rows = []
    order = np.argsort(-point)
    for j in order:
        rows.append({
            "feature": feats[j],
            "share": round(float(point[j]), 4),
            "ci_low": round(float(lo[j]), 4),
            "ci_high": round(float(hi[j]), 4),
            "ci_width": round(float(hi[j] - lo[j]), 4),
            f"top{top_k}_stability": round(float(stability[j]), 3),
        })

    unstable = [r for r in rows[:top_k] if r[f"top{top_k}_stability"] < 0.9]
    touching_zero = [r for r in rows[:top_k] if r["ci_low"] <= 0.001]
    if not unstable and not touching_zero:
        verdict = (
            f"STABLE: top {top_k} features keep their ranks in at least 90% "
            f"of {n_bootstrap} replicates and no interval touches zero"
        )
    else:
        parts = []
        if touching_zero:
            parts.append(
                "intervals touching zero: "
                + ", ".join(r["feature"] for r in touching_zero)
            )
        if unstable:
            parts.append(
                "rank unstable: "
                + ", ".join(f"{r['feature']} ({r[f'top{top_k}_stability']:.0%})"
                            for r in unstable)
            )
        verdict = "WEAK EVIDENCE: " + "; ".join(parts)

    return {
        "n_bootstrap": n_bootstrap,
        "rows_per_replicate": m,
        "features": rows,
        "verdict": verdict,
        "method": "bootstrap_percentile_shap",
        "reference": "Efron bootstrap percentile intervals; Goldwasser et al. (2024) significance of importance rankings",
        "note": (
            "shares are mean absolute SHAP normalized per replicate; "
            "rank stability is the fraction of replicates within the top k"
        ),
    }
