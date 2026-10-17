"""Counterfactual fairness auditing.

Checks whether a model would give the same individual a different outcome
in a counterfactual world where only a protected attribute changed.

Reference: Kusner et al. "Counterfactual Fairness" (NeurIPS 2017),
"Counterfactual Fairness: Removing Influence of Sensitive Attributes".

Uses CausalLens's existing intervene() infrastructure.
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
        driver: Any CausalLens driver with intervene() support.
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
