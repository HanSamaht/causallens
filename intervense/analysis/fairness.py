"""Counterfactual fairness auditing.

Checks whether a model would give the same individual a different outcome
in a counterfactual world where only a protected attribute changed.

Reference: Kusner et al. "Counterfactual Fairness" (NeurIPS 2017),
"Counterfactual Fairness: Removing Influence of Sensitive Attributes".

Uses Intervense's existing intervene() infrastructure.
"""

from __future__ import annotations

import numpy as np


def counterfactual_fairness_check(
    driver,
    sample_indices: list[int],
    protected_feature: str,
    contrast_values: list[float],
    threshold: float = 0.5,
) -> dict:
    """Check if changing only the protected attribute flips predictions.

    For each sample, intervenes on the protected feature to each contrast
    value while holding all other features fixed, then measures whether
    the prediction crosses the decision threshold.

    A prediction flip indicates a counterfactual fairness violation:
    the same individual receives different outcomes based solely on the
    protected attribute.

    Args:
        driver: Any Intervense driver with intervene() support.
        sample_indices: Background sample indices to test.
        protected_feature: Feature name (e.g. "person_age", "sex").
        contrast_values: Values to set (e.g. [25.0, 45.0, 65.0]).
        threshold: Decision boundary (0.5 for binary classification).

    Returns:
        Dict with per-sample results and aggregate fairness metrics.
    """
    results = []
    flips = 0
    total = 0

    for idx in sample_indices:
        x = driver.sample_at(idx)
        p_base = driver.predict_one(x)

        for cv in contrast_values:
            x_cf = driver.intervene(x, {protected_feature: cv})
            p_cf = driver.predict_one(x_cf)
            base_label = 1 if p_base > threshold else 0
            cf_label = 1 if p_cf > threshold else 0
            flipped = base_label != cf_label
            total += 1
            flips += int(flipped)

            results.append({
                "sample_index": idx,
                "base_prediction": round(float(p_base), 4),
                "base_label": base_label,
                "cf_value": cv,
                "cf_prediction": round(float(p_cf), 4),
                "cf_label": cf_label,
                "flipped": flipped,
                "prediction_shift": round(float(p_cf - p_base), 4),
            })

    flip_rate = flips / total if total > 0 else 0.0
    avg_shift = (
        np.mean([abs(r["prediction_shift"]) for r in results]) if results else 0.0
    )

    if flip_rate == 0.0 and avg_shift < 0.01:
        verdict = "PASS: no counterfactual fairness violation detected"
    elif flip_rate > 0.1:
        verdict = (
            f"VIOLATION: {flips}/{total} predictions flip when only "
            f"'{protected_feature}' changes. Same individuals receive "
            f"different outcomes based solely on this attribute."
        )
    else:
        verdict = (
            f"WARNING: {flips}/{total} prediction flips. Low but non-zero "
            f"counterfactual unfairness. Average shift: {avg_shift:.4f}."
        )

    return {
        "protected_feature": protected_feature,
        "contrast_values": contrast_values,
        "n_samples": len(sample_indices),
        "n_tests": total,
        "flip_count": flips,
        "flip_rate": round(flip_rate, 4),
        "avg_prediction_shift": round(float(avg_shift), 4),
        "verdict": verdict,
        "per_sample": results,
        "method": "counterfactual_invariance",
        "reference": "Kusner et al. Counterfactual Fairness (NeurIPS 2017)",
    }


def fairness_path_decomposition(
    driver,
    sample_indices: list[int],
    protected_feature: str,
    contrast_low: float,
    contrast_high: float,
    mediators: list[str] | None = None,
    max_mediators: int = 3,
    mediator_correlation: float = 0.3,
) -> dict:
    """Decompose the protected attribute effect into direct and indirect paths.

    The plain counterfactual check measures the total effect of the
    protected attribute. Fairness practice needs to know which path
    carries it:

      direct path:   protected -> prediction, bypassing all mediators
      indirect path: protected -> mediator -> prediction, per mediator

    An effect that rides only through a legitimate mediator (income,
    employment) is a different legal situation from a direct effect of
    the protected attribute on the decision.

    Estimation per sample, in the natural effects framework (Pearl 2001)
    adapted to input features. Because intervening on the protected
    attribute does not regenerate the mediator features from the data,
    each mediator mechanism is learned as a linear model m ~ A over the
    background, and per-sample natural values are built from the fitted
    coefficients plus the sample residual:

      x_lo  = do(A := low, M := natural low-regime values)
      x_hi  = do(A := high, M := natural high-regime values)
      TE    = p(x_hi) - p(x_lo)
      direct = p(do(A := high, M := m_lo)) - p(x_lo)
      indirect(m) = p(do(A := low, m := m_hi(m), other mediators at m_lo))
                    - p(x_lo)

    Mediators are auto-detected as the features most correlated with the
    protected attribute unless an explicit list is given.

    Returns aggregate means, per path shares, and a verdict that names
    direct versus indirect discrimination.
    """
    feats = driver.feature_names
    if protected_feature not in feats:
        raise ValueError(f"unknown feature {protected_feature!r}")
    if mediators is None:
        bg = driver.background
        a_col = bg[:, feats.index(protected_feature)]
        scored = []
        for f in feats:
            if f == protected_feature:
                continue
            c = abs(float(np.corrcoef(a_col, bg[:, feats.index(f)])[0, 1]))
            if c >= mediator_correlation:
                scored.append((c, f))
        scored.sort(reverse=True)
        mediators = [f for _, f in scored[:max_mediators]]
    else:
        for m in mediators:
            if m not in feats:
                raise ValueError(f"unknown mediator {m!r}")

    # linear mediator mechanisms m ~ 1 + A over the background
    bg = driver.background
    a_col = bg[:, feats.index(protected_feature)]
    A = np.column_stack([np.ones(len(a_col)), a_col])
    coef = {m: np.linalg.lstsq(A, bg[:, feats.index(m)], rcond=None)[0] for m in mediators}

    def natural_values(x, regime_value: float) -> dict:
        """Mediator values the causal mechanism produces under do(A := regime_value)."""
        out = {}
        for m in mediators:
            b0, b1 = coef[m]
            observed = float(x[feats.index(m)])
            residual = observed - (b0 + b1 * float(x[feats.index(protected_feature)]))
            out[m] = b0 + b1 * regime_value + residual
        return out

    tes, directs = [], []
    indirect_runs = {m: [] for m in mediators}
    per_sample = []

    for idx in sample_indices:
        x = driver.sample_at(idx)
        m_lo_vals = natural_values(x, contrast_low)
        m_hi_vals = natural_values(x, contrast_high)

        # both regimes carry natural mediator values so the decomposition
        # is not contaminated by observed versus fitted mediator drift
        lo_plan = {protected_feature: contrast_low} | m_lo_vals
        hi_plan = {protected_feature: contrast_high} | m_hi_vals
        p_lo = driver.predict_one(driver.intervene(x, lo_plan))
        p_hi = driver.predict_one(driver.intervene(x, hi_plan))
        te = float(p_hi - p_lo)
        tes.append(te)

        # direct path: change A, freeze every mediator at its low-regime value
        plan = {protected_feature: contrast_high} | m_lo_vals
        p_direct = driver.predict_one(driver.intervene(x, plan))
        directs.append(float(p_direct - p_lo))

        indirect_sample = {}
        for m in mediators:
            ind_plan = {protected_feature: contrast_low}
            ind_plan.update({mm: m_hi_vals[m] if mm == m else m_lo_vals[mm]
                             for mm in mediators})
            p_ind = driver.predict_one(driver.intervene(x, ind_plan))
            eff = float(p_ind - p_lo)
            indirect_runs[m].append(eff)
            indirect_sample[m] = round(eff, 6)

        per_sample.append({
            "sample_index": idx,
            "p_low": round(float(p_lo), 6),
            "p_high": round(float(p_hi), 6),
            "total_effect": round(te, 6),
            "direct_effect": round(directs[-1], 6),
            "indirect_effects": indirect_sample,
        })

    n = max(len(tes), 1)
    mean_te = float(np.mean(tes))
    mean_direct = float(np.mean(directs))
    mean_indirect = {m: float(np.mean(v)) for m, v in indirect_runs.items()}

    scale = abs(mean_te) if abs(mean_te) > 1e-9 else 1.0
    direct_share = mean_direct / scale
    indirect_shares = {m: v / scale for m, v in mean_indirect.items()}

    paths = {
        "direct": {
            "mean_effect": round(mean_direct, 6),
            "share": round(float(direct_share), 4),
        }
    }
    for m in mediators:
        paths[f"via {m}"] = {
            "mean_effect": round(mean_indirect[m], 6),
            "share": round(float(indirect_shares[m]), 4),
        }

    top_indirect = max(mean_indirect, key=lambda m: abs(mean_indirect[m])) if mediators else None
    if abs(mean_te) < 0.01:
        verdict = (
            f"NEGLIGIBLE: total effect of '{protected_feature}' is {mean_te:+.4f}, "
            "below the materiality bar of 0.01"
        )
    elif abs(direct_share) >= 0.5:
        verdict = (
            f"DIRECT DISCRIMINATION RISK: {abs(direct_share):.0%} of the "
            f"'{protected_feature}' effect reaches the prediction without "
            "passing through any measured mediator"
        )
    elif top_indirect is not None:
        verdict = (
            f"INDIRECT PATH: the '{protected_feature}' effect flows mainly "
            f"through {top_indirect} (share {abs(indirect_shares[top_indirect]):.0%}). "
            "Review whether that mediator is a legitimate decision factor"
        )
    else:
        verdict = "NO MEDIATED PATH EXPLAINS THE EFFECT"

    return {
        "protected_feature": protected_feature,
        "contrast": {"low": contrast_low, "high": contrast_high},
        "mediators": mediators,
        "n_samples": len(sample_indices),
        "mean_total_effect": round(mean_te, 6),
        "paths": paths,
        "verdict": verdict,
        "per_sample": per_sample,
        "method": "path_decomposition_natural_effects",
        "reference": "Pearl (2001) Direct and Indirect Effects",
        "note": (
            "shares are computed against the total effect and may not sum to 1 "
            "when path interactions exist"
        ),
    }
