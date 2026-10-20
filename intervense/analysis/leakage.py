"""Leakage and contamination checks: no lookahead in the model, no test
rows hiding in training.

Three tools:
  1. duplicate_overlap between training and test sets:
     test samples present in training means contamination
  2. single_feature_leak_scan:
     a single feature with AUC close to the model AUC means leakage,
     though a strong legitimate feature can look the same, so verify
     before deleting
  3. label_distribution_check for training versus test drift

Known gap, GAP-002 in the bug registry: leakage flags on a driver
background cannot be confirmed by do(feature := one bar) effects yet;
the window level driver support is still missing.
"""
from __future__ import annotations

import numpy as np


def duplicate_overlap(X_train, X_test, near_tol: float = 0.01, chunk: int = 256) -> dict:
    """: + min < near_tol."""
    A = np.asarray(X_train, dtype=float)
    B = np.asarray(X_test, dtype=float)
    rng_vec = np.maximum(A.max(axis=0) - A.min(axis=0), 1e-12)
    An, Bn = A / rng_vec, B / rng_vec

    exact = 0
    near = 0
    suspect_idx: list[int] = []
    for i in range(0, len(Bn), chunk):
        blk = Bn[i : i + chunk] # (c, F)
        d = np.abs(blk[:, None, :] - An[None, :, :]).sum(axis=2) # (c, n_train)
        dmin = d.min(axis=1)
        exact += int((dmin < 1e-9).sum())
        near += int((dmin < near_tol).sum())
        suspect_idx += (i + np.where(dmin < near_tol)[0]).tolist()
    n = len(Bn)
    near_rate = (exact + near) / n if n else 0.0
    verdict = (
        "contamination" if exact / max(n, 1) > 0.02
        else (":,check" if near_rate > 0.15
              else (":sample" if near_rate > 0.02 else ":valid"))
    )
    return {
        "n_test": n,
        "exact_duplicates": exact,
        "near_duplicates_total": near,
        "overlap_rate": round(float(near_rate), 4),
        "suspect_test_indices": suspect_idx[:20],
        "verdict": verdict,
        "note": "=L1<%.3f;inbar," % near_tol,
    }


def single_feature_leak_scan(y, X, feature_names, model_auc: float | None = None) -> dict:
    """feature solo AUC:leakagefeature."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    X = np.asarray(X, dtype=float)
    rows = []
    for j, f in enumerate(feature_names):
        col = X[:, j]
        m = ~np.isnan(col)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            continue
        try:
            auc = max(
                roc_auc_score(y[m], col[m]),
                roc_auc_score(y[m], -col[m]),
            )
        except Exception:
            continue
        rows.append({"feature": f, "solo_auc": round(float(auc), 4)})
    rows.sort(key=lambda r: -r["solo_auc"])
    flagged = [r for r in rows if r["solo_auc"] >= 0.75]
    borderline = [r for r in rows if 0.65 <= r["solo_auc"] < 0.75]
    return {
        "top_features": rows[:8],
        "leak_suspects": flagged,
        "borderline": borderline,
        "model_auc_reference": model_auc,
        "verdict": (
            f"{len(flagged)} features with solo AUC at or above 0.75, verify each before removal"
            if flagged
            else f"{len(borderline)} features between 0.65 and 0.75, treat as borderline"
        ),
        "note": "solo AUC 0.6-0.7 normal;>0.75 on'notthethis'featureleakageverify",
    }


def label_distribution_check(y_train, y_test) -> dict:
    """Label distribution check."""
    a, b = np.asarray(y_train, dtype=float), np.asarray(y_test, dtype=float)
    pos_tr, pos_te = float((a > 0).mean()), float((b > 0).mean())
    gap = abs(pos_te - pos_tr)
    return {
        "train_positive_rate": round(pos_tr, 4),
        "test_positive_rate": round(pos_te, 4),
        "gap": round(gap, 4),
        "verdict": "consistent" if gap < 0.1 else "driftordifference,compareconclusion",
    }
