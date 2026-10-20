"""Double machine learning: cross-fitted partially linear effects.

The framework's intervention tools estimate what the MODEL does. This
module estimates what the DATA supports when the treatment is not
randomly assigned: the confounded observational regime every real
analysis lives in.

Robinson's partially linear model: y = theta * T + g(X) + noise, with
T = m(X) + own noise. Cross-fitted double ML removes both nuisance
functions with any learner (the driver's own family), then estimates
theta on the residual-on-residual regression with robust inference.
Cross-fitting (nuisances fit on other folds, residuals predicted on
held out folds) removes the own-observation overfitting bias that
naive residualization carries.

Assumptions, disclosed rather than hidden: the effect is linear and
constant in the treatment (use cate_heterogeneity to check the second
one), and the treatment is unconfounded GIVEN the observed features.
The verdict says what the data licenses, not what is true.
"""
from __future__ import annotations

import numpy as np


def double_ml_effect(
    driver,
    treatment: str,
    y,
    n_folds: int = 5,
    sample_weight=None,
    seed: int = 0,
) -> dict:
    """Cross-fitted partially linear treatment effect with robust SEs.

    Args:
        driver: any CausalLens driver whose model is a cloneable
            sklearn estimator or pipeline (used as both nuisance
            learners).
        treatment: the treatment column name.
        y: outcome vector aligned with driver.background.
        n_folds: cross-fitting folds, at least 2.
        sample_weight: per-row weights for nuisance fits.
        seed: reproducibility.

    Returns:
        Dict with theta per treatment unit and per p10 to p90
        contrast, robust standard error and 95 percent interval, the
        naive OLS estimate for comparison (the confounding gap), and
        a SIGNIFICANT or NOT SIGNIFICANT verdict.
    """
    from sklearn.base import clone
    from sklearn.linear_model import LinearRegression

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    feats = list(driver.feature_names)
    if treatment not in feats:
        raise ValueError(f"unknown treatment: {treatment}")
    X = np.asarray(driver.background, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if len(y) != len(X):
        raise ValueError(
            f"y has {len(y)} rows but the background has {len(X)}")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError(
            "NaN in background or y; impute before calling "
            "double_ml_effect")

    t_idx = feats.index(treatment)
    T = X[:, t_idx].astype(float)
    controls = [i for i in range(len(feats)) if i != t_idx]
    Xc = X[:, controls]
    sw = (None if sample_weight is None
          else np.asarray(sample_weight, dtype=float))

    try:
        probe = clone(driver.model)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            "double_ml_effect uses the driver's model family as the "
            f"nuisance learner and sklearn could not clone it "
            f"({type(e).__name__}: {e})") from e

    rng = np.random.default_rng(seed)
    folds = np.arange(len(X)) % n_folds
    rng.shuffle(folds)

    res_y = np.full(len(X), np.nan)
    res_t = np.full(len(X), np.nan)
    for k in range(n_folds):
        te = folds == k
        tr = ~te
        # outcome nuisance: E[y | X controls]
        my = clone(driver.model)
        # treatment nuisance: E[T | X controls]
        mt = clone(driver.model)
        if sw is None:
            my.fit(Xc[tr], y[tr])
            mt.fit(Xc[tr], T[tr])
        else:
            my.fit(Xc[tr], y[tr], sample_weight=sw[tr])
            mt.fit(Xc[tr], T[tr], sample_weight=sw[tr])
        res_y[te] = y[te] - my.predict(Xc[te])
        res_t[te] = T[te] - mt.predict(Xc[te])

    denom = float(np.sum(res_t ** 2))
    if denom < 1e-12:
        raise ValueError(
            "treatment is fully explained by the controls; no "
            "exogenous variation to estimate from")
    theta = float(np.sum(res_t * res_y) / denom)

    # robust (Neyman-orthogonal) score and standard error
    score = res_t * (res_y - theta * res_t)
    n = len(X)
    var = float(np.sum((score - score.mean()) ** 2) / n) / (denom / n) ** 2
    se = float(np.sqrt(var / n))
    lo, hi = theta - 1.96 * se, theta + 1.96 * se

    # naive OLS with and without controls, for the confounding gap
    def ols_beta(cols):
        M = np.column_stack([X[:, cols], np.ones(n)])
        return float(np.linalg.lstsq(M, y, rcond=None)[0][0])

    naive_raw = ols_beta([t_idx])
    naive_ctrl = ols_beta([t_idx] + controls)

    q10 = float(np.percentile(T, 10))
    q90 = float(np.percentile(T, 90))
    width = q90 - q10

    if lo > 0 or hi < 0:
        verdict = (
            f"SIGNIFICANT: theta {theta:+.4f} per treatment unit, "
            f"95% CI [{lo:+.4f}, {hi:+.4f}] excludes zero"
        )
    else:
        verdict = (
            f"NOT SIGNIFICANT: theta {theta:+.4f}, 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] crosses zero"
        )

    return {
        "treatment": treatment,
        "n_rows": n,
        "n_folds": n_folds,
        "theta_per_unit": round(theta, 6),
        "theta_per_p10_p90": round(theta * width, 6),
        "contrast": {"from": round(q10, 6), "to": round(q90, 6)},
        "se_robust": round(se, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "naive_ols_uncontrolled": round(naive_raw, 6),
        "naive_ols_controlled": round(naive_ctrl, 6),
        "confounding_gap_raw": round(naive_raw - theta, 6),
        "confounding_gap_controlled": round(naive_ctrl - theta, 6),
        "verdict": verdict,
        "method": "cross_fitted_double_ml_plr",
        "reference": (
            "Chernozhukov et al. (2018) Double/debiased machine "
            "learning; Robinson (1988) partially linear model"
        ),
        "note": (
            "assumes a linear constant treatment effect "
            "(check with cate_heterogeneity) and unconfoundedness "
            "given observed features (stress with "
            "omitted_variable_sensitivity); nuisance learners are the "
            "driver's own family, cross-fitted"
        ),
    }
