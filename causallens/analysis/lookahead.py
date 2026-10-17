"""lookahead(look-ahead):featurein"alsonotin".

method(,intervention, lookahead-bias :
,,value):

  IC —— everyfeaturecomparebelow:
    IC_now = corr( f_t, y_t ) feature"in"has
    IC_prev = corr( f_{t-k}, y_t ) feature k to
  feature:IC (that,not)
  lookaheadfeature:IC_now ,IC_prev -> onlyin" bar"in,
  featureto(or bar )

  value:leakagefeature = |IC_now| ( Sharpe ).
"""
from __future__ import annotations

import numpy as np


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 20:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def lookahead_scan(y, X_now, X_prev, feature_names, collapse_ratio: float = 0.35) -> dict:
    """X_now: t feature;X_prev: sample,feature t-1(or t-k)value.
    (/), X_prev().
    """
    y = np.asarray(y, dtype=float)
    A, B = np.asarray(X_now, dtype=float), np.asarray(X_prev, dtype=float)
    rows = []
    for j, f in enumerate(feature_names):
        ic_now = _rank_ic(A[:, j], y)
        ic_prev = _rank_ic(B[:, j], y)
        if np.isnan(ic_now):
            continue
        ratio = ic_prev / ic_now if abs(ic_now) > 1e-6 else 1.0
        rows.append({
            "feature": f,
            "ic_now": round(ic_now, 4),
            "ic_lagged": round(ic_prev, 4) if not np.isnan(ic_prev) else None,
            "survival_ratio": round(float(ratio), 3),
        })
    rows.sort(key=lambda r: -abs(r["ic_now"]))
    suspects = [
        r for r in rows
        if abs(r["ic_now"]) >= 0.03 and r["survival_ratio"] is not None
        and r["survival_ratio"] < collapse_ratio
    ]
    return {
        "top_features": rows[:8],
        "leak_suspects": suspects,
        "verdict": (
            f"finding {len(suspects)} lookaheadfeature(IC in)——verifyitsis"
            if suspects
            else "lookaheadfeature:allfeature IC (normal)"
        ),
        "note": ":onlyfeatureand,notmodel; leak_scan(featuresolo AUC)more",
    }
