"""Diverse counterfactual search (DiCE-style).

Generates multiple distinct counterfactual paths instead of one.
Each path uses a different feature subset, giving users options.
"""
from __future__ import annotations

import numpy as np

from .attribution import explain_sample
from .counterfactual import _base_state, _eval_flip, _make_result, default_threshold


def diverse_counterfactuals(driver, x, pack=None, n_paths=3, max_features=3,
                            threshold=None, seed=0) -> list:
    """Find multiple distinct counterfactual paths.

    Each path uses a different combination of features, so users see
    all their options: "reduce loan amount" vs "extend employment" etc.
    """
    x = np.asarray(x, dtype=float).ravel()
    threshold = threshold or default_threshold(driver)
    p_base, base_label, thr = _base_state(driver, x)

    ranked = [a.feature for a in explain_sample(driver, x)]
    rng = np.random.default_rng(seed)

    quantiles = (0.05, 0.25, 0.75, 0.90, 0.95)
    candidates = []
    for f in ranked[:10]:
        for q in quantiles:
            assignments = {f: driver.quantile_of(f, q)}
            flipped, p, x2 = _eval_flip(driver, x, assignments, p_base, base_label, thr)
            if flipped:
                candidates.append((f, q, assignments, p))

    # Build diverse paths: different feature subsets
    paths = []
    used_features = set()

    # Sort candidates by flip magnitude
    candidates.sort(key=lambda c: abs(c[3] - p_base), reverse=True)

    for f, q, assignments, p in candidates:
        if f not in used_features:
            paths.append({
                "changed_features": {f: assignments[f]},
                "prediction_after": round(float(p), 4),
                "flipped": True,
                "method": "diverse_single",
            })
            used_features.add(f)
        if len(paths) >= n_paths:
            break

    # Try pair combinations for additional paths if needed
    if len(paths) < n_paths:
        top = ranked[:6]
        for i, f1 in enumerate(top):
            for f2 in top[i+1:]:
                if {f1, f2} & used_features:
                    continue
                for q1 in (0.1, 0.9):
                    for q2 in (0.1, 0.9):
                        assignments = {f1: driver.quantile_of(f1, q1),
                                      f2: driver.quantile_of(f2, q2)}
                        flipped, p, x2 = _eval_flip(driver, x, assignments, p_base, base_label, thr)
                        if flipped:
                            paths.append({
                                "changed_features": assignments,
                                "prediction_after": round(float(p), 4),
                                "flipped": True,
                                "method": "diverse_pair",
                            })
                            used_features.update([f1, f2])
                            break
                    if len(paths) >= n_paths:
                        break
            if len(paths) >= n_paths:
                break

    return paths
