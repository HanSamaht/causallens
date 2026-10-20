"""Omitted variable sensitivity: how afraid of the unobserved should you be.

Every controlled estimate carries the same silent assumption: the
controls captured everything that matters. This module quantifies the
cost of being wrong about that, in the tradition of Oster (2019):
compare the coefficient before and after controls, together with how
much variance the controls explained, and report delta, the ratio of
selection on unobservables to selection on observables that would
nullify the effect.

Reading the verdict:

  delta above 1: unobservables would need to matter MORE than the
  observables already controlled for; the finding is robust in this
  sense
  delta below 1: a weaker unobservable than the controls you included
  could kill the finding; fragile

Also reported: the bias-adjusted beta at delta = 1 and at the
commonly used R-squared max of 1.3 times the full model's R-squared.
The module is linear-model based, like the specification audit's
certificate, and says so.
"""
from __future__ import annotations

import numpy as np


def _ols(y, X):
    M = np.column_stack([X, np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(M, y, rcond=None)
    pred = M @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(beta[0]), r2


def omitted_variable_sensitivity(
    driver,
    treatment: str,
    y,
    r_max_factor: float = 1.3,
) -> dict:
    """Oster delta and bias-adjusted bounds for a controlled effect.

    Args:
        driver: any CausalLens driver (only background and
            feature_names are read; the estimate is linear by design).
        treatment: feature whose coefficient is stress tested.
        y: outcome aligned with driver.background.
        r_max_factor: the assumed R-squared ceiling as a multiple of
            the full model's R-squared, capped at 1. Oster's
            convention is 1.3.

    Returns:
        Dict with beta_uncontrolled and beta_controlled, both
        R-squareds, delta (the unobservable strength that nullifies
        the effect), bias-adjusted beta at delta 1, and a ROBUST or
        FRAGILE verdict.
    """
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
            "omitted_variable_sensitivity")

    t_idx = feats.index(treatment)
    controls = [i for i in range(len(feats)) if i != t_idx]

    # standardize y and treatment so betas read in SD units
    zy = (y - y.mean()) / (y.std() + 1e-12)
    zt = (X[:, t_idx] - X[:, t_idx].mean()) / (X[:, t_idx].std() + 1e-12)

    beta_r, r2_r = _ols(zy, zt)
    Zc = (X[:, controls] - X[:, controls].mean(axis=0)) / (
        X[:, controls].std(axis=0) + 1e-12)
    beta_f, r2_f = _ols(zy, np.column_stack([zt, Zc]))

    r2_max = min(1.0, r_max_factor * r2_f)

    denom_gain = r2_f - r2_r
    coef_move = beta_r - beta_f

    if abs(coef_move) < 1e-9 or denom_gain < 1e-9:
        verdict = (
            "INCONCLUSIVE: controls barely move the coefficient or add "
            "no explanatory power, delta is not identified; report the "
            f"stability itself (beta {beta_r:+.4f} to {beta_f:+.4f})"
        )
        return {
            "treatment": treatment,
            "beta_uncontrolled": round(beta_r, 6),
            "beta_controlled": round(beta_f, 6),
            "r2_uncontrolled": round(r2_r, 6),
            "r2_full": round(r2_f, 6),
            "delta": None,
            "beta_adjusted_delta1": None,
            "verdict": verdict,
            "method": "oster_delta",
            "reference": "Oster (2019) Journal of Business and Economic "
                         "Statistics, coefficient stability and bias",
            "note": (
                "linear in standardized units; delta is the ratio of "
                "unobservable to observable selection that drives the "
                "effect to zero"
            ),
        }

    beta_star_1 = beta_f - coef_move * (r2_max - r2_f) / denom_gain
    delta_star = beta_f * denom_gain / (coef_move * (r2_max - r2_f))

    if delta_star < 0:
        # controls moved the coefficient away from zero; the selection
        # bound does not apply in the attenuation direction and the
        # honest verdict is that the framework cannot certify this one
        return {
            "treatment": treatment,
            "n_rows": len(X),
            "beta_uncontrolled": round(beta_r, 6),
            "beta_controlled": round(beta_f, 6),
            "r2_uncontrolled": round(r2_r, 6),
            "r2_full": round(r2_f, 6),
            "r2_max": round(r2_max, 6),
            "delta": round(float(delta_star), 4),
            "beta_adjusted_delta1": round(float(beta_star_1), 6),
            "verdict": (
                "INCONCLUSIVE (negative delta): controls moved the "
                "coefficient away from zero, the attenuation bound does "
                "not apply; treat the estimate as sensitive to omitted "
                "variables of the same character as the controls"
            ),
            "method": "oster_delta",
            "reference": (
                "Oster (2019) Journal of Business and Economic "
                "Statistics, coefficient stability and bias"
            ),
            "note": (
                "linear in standardized units; delta is the ratio of "
                "unobservable to observable selection that drives the "
                "effect to zero; r_max is the 1.3 R-squared convention"
            ),
        }

    robust = abs(delta_star) > 1 and np.sign(beta_star_1) == np.sign(beta_f)
    if robust:
        verdict = (
            f"ROBUST: delta {delta_star:+.2f}, unobservables would need "
            f"{abs(delta_star):.2f}x the selection of the controls to "
            f"nullify the effect; bias-adjusted beta at delta 1 is "
            f"{beta_star_1:+.4f}, same sign as the estimate"
        )
    else:
        verdict = (
            f"FRAGILE: delta {delta_star:+.2f}, an unobservable weaker "
            f"than the included controls ({abs(delta_star):.2f}x) could "
            f"nullify the effect; bias-adjusted beta at delta 1 is "
            f"{beta_star_1:+.4f}"
        )

    return {
        "treatment": treatment,
        "n_rows": len(X),
        "beta_uncontrolled": round(beta_r, 6),
        "beta_controlled": round(beta_f, 6),
        "r2_uncontrolled": round(r2_r, 6),
        "r2_full": round(r2_f, 6),
        "r2_max": round(r2_max, 6),
        "delta": round(float(delta_star), 4),
        "beta_adjusted_delta1": round(float(beta_star_1), 6),
        "verdict": verdict,
        "method": "oster_delta",
        "reference": (
            "Oster (2019) Journal of Business and Economic Statistics, "
            "coefficient stability and bias"
        ),
        "note": (
            "linear in standardized units; delta is the ratio of "
            "unobservable to observable selection that drives the "
            "effect to zero; r_max is the 1.3 R-squared convention"
        ),
    }
