"""Panel fixed effects with cluster-robust inference.

The within-player analysis pattern that consumed our NFL writeup:
same unit observed many times, the question is whether within-unit
variation in X moves Y, with stable unit quality absorbed and
inference that respects the clustering. This module is that pattern
as one call: within-transformation by group, OLS on the demeaned
data, and the cluster-robust sandwich (Stata-style finite sample
correction) so p-values stop flattering the number of rows.

The verdict names the significant within-unit effects in probability
or outcome units per +1 SD and states the cluster count, because a
fixed effect estimate with six clusters is a rumor, not a finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cluster_vcov(X: np.ndarray, resid: np.ndarray,
                  groups: np.ndarray) -> np.ndarray:
    """Sandwich covariance with Stata's finite sample correction."""
    XtX_inv = np.linalg.inv(X.T @ X)
    unique = np.unique(groups)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in unique:
        m = groups == g
        Xg = X[m] * resid[m][:, None]
        meat += Xg.T @ Xg
    g_, n_, k_ = len(unique), len(X), X.shape[1]
    corr = (g_ / max(g_ - 1, 1)) * ((n_ - 1) / max(n_ - k_, 1))
    return corr * (XtX_inv @ meat @ XtX_inv)


def panel_fixed_effects(
    df: pd.DataFrame,
    outcome: str,
    features: list[str],
    group: str,
    cluster: str | None = None,
    min_groups: int = 10,
) -> dict:
    """Within-group fixed effects with cluster-robust standard errors.

    Args:
        df: one row per observation, panel or repeated measures.
        outcome: column name of the dependent variable.
        features: within-group varying predictors.
        group: unit identifier for the fixed effect (demeaning).
        cluster: cluster variable for robust SEs, defaults to group.
        min_groups: below this the verdict warns that inference is a
            rumor regardless of row count.

    Returns:
        Dict with per-feature coefficient, cluster-robust SE, p value,
        and 95 percent interval, plus the naive (non-clustered) SE for
        comparison and one verdict.
    """
    if outcome not in df.columns or group not in df.columns:
        raise ValueError("outcome and group must be columns of df")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"unknown feature columns: {missing}")
    cluster_col = cluster or group
    select = [outcome] + features + [group, cluster_col]
    # group==cluster selects the same column twice; deduplicate
    select = list(dict.fromkeys(select))
    d = df[select].dropna().copy()
    for c in [outcome] + features:
        d[c] = d[c].astype(float)
    n_groups = len(d[group].unique())
    if n_groups < 2:
        raise ValueError("need at least two groups to demean")

    # standardize features so coefficients read per +1 SD
    for c in features:
        d[c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-12)
    y_mean = d[outcome].mean()
    y_sd = d[outcome].std() + 1e-12
    d[outcome] = (d[outcome] - y_mean) / y_sd

    # within transformation per group
    w = d.groupby(group)[[outcome] + features].transform(
        lambda s: s - s.mean())
    X = w[features].to_numpy()
    y = w[outcome].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    groups = d[cluster_col].to_numpy()
    vcov = _cluster_vcov(X, resid, groups)
    se_cluster = np.sqrt(np.diag(vcov))
    XtX_inv = np.linalg.inv(X.T @ X)
    se_naive = np.sqrt(np.diag(XtX_inv) * (resid @ resid)
                       / max(len(X) - X.shape[1], 1))

    rows = []
    sig = []
    for i, f in enumerate(features):
        b = float(beta[i])
        se = float(se_cluster[i])
        p = float(2 * (1 - _norm_cdf(abs(b / se)))) if se > 0 else 0.0
        lo, hi = b - 1.96 * se, b + 1.96 * se
        significant = p < 0.05
        if significant:
            sig.append(f"{f} {b:+.3f} (p {p:.3g})")
        rows.append({
            "feature": f,
            "coef_within_per_sd": round(b, 6),
            "se_cluster": round(se, 6),
            "se_naive": round(float(se_naive[i]), 6),
            "p_value": round(p, 6),
            "ci95": [round(lo, 6), round(hi, 6)],
            "significant": significant,
        })

    if not sig:
        verdict = (
            f"NO WITHIN-EFFECTS: none of {len(features)} features clear "
            "5 percent with cluster-robust inference"
        )
    elif n_groups < min_groups:
        verdict = (
            f"SIGNIFICANT BUT FRAGILE: {', '.join(sig)}; only "
            f"{n_groups} clusters, treat as suggestive"
        )
    else:
        verdict = (
            f"WITHIN-EFFECTS: {', '.join(sig)} across {n_groups} "
            f"clusters, {len(d)} rows"
        )

    return {
        "n_rows": len(d),
        "n_groups": int(n_groups),
        "features": rows,
        "verdict": verdict,
        "method": "within_transformation_ols_cluster_sandwich",
        "reference": (
            "Arellano (1987) cluster covariance; Cameron and Miller "
            "(2015) practitioner's guide to cluster-robust inference"
        ),
        "note": (
            "coefficients are per +1 SD of the feature on the "
            "standardized outcome, within group; stable unit quality is "
            "absorbed by demeaning; the naive SE is printed so the "
            "clustering penalty is visible"
        ),
    }


def _norm_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2.0)))
