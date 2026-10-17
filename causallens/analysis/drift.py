"""drift(:KS + PSI + predictiondistributiondrift).

distributiondrift SHAP ——"backgrounddistributionalso".
not evidently/alibi-detect (optionalgenerate HTML report),
KS/PSI ,andtest.

  - KS:sample Kolmogorov-Smirnov ,α=0.05 value 1.36*sqrt((n+m)/(nm))
  - PSI:Population Stability Index,>0.25 significantdrift()
  - predictiondrift:modeloutputdistributionin/on KS
"""
from __future__ import annotations

import numpy as np

from ..drivers.base import MicDriver


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """sample KS D()."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ks_pvalue_approx(d: float, n: int, m: int) -> float:
    """ p value(Smirnov ),infer."""
    ne = n * m / (n + m)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return float(max(0.0, min(1.0, p)))


def _psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """PSI:bydistributionpercentile(percentileeveryhassample)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def check_drift(driver: MicDriver, new_data: np.ndarray, alpha: float = 0.05) -> dict:
    """background vs featuredrift + predictiondistributiondrift.

    return overall_verdict: stable / drifting / severely_drifting.
    driftby,"when SHAP distribution,conclusion".
    """
    ref = driver.background
    new = np.asarray(new_data, dtype=float)
    assert new.shape[1] == ref.shape[1], "andbackgroundnotconsistent"

    n_feats = []
    for j, f in enumerate(driver.feature_names):
        d = _ks_statistic(ref[:, j], new[:, j])
        p = _ks_pvalue_approx(d, len(ref), len(new))
        psi = _psi(ref[:, j], new[:, j])
        drifted = p < alpha or psi > 0.25
        n_feats.append(
            {
                "feature": f,
                "ks": round(d, 4),
                "ks_pvalue": round(p, 4),
                "psi": round(psi, 4),
                "drifted": bool(drifted),
            }
        )

    p_ref = driver.predict(ref)
    p_new = driver.predict(new)
    pred_ks = _ks_statistic(p_ref, p_new)
    pred_drift = pred_ks > 1.36 * np.sqrt((len(ref) + len(new)) / (len(ref) * len(new)))

    n_drift = sum(x["drifted"] for x in n_feats)
    share = n_drift / len(n_feats)
    verdict = "severely_drifting" if share >= 0.3 or pred_drift else ("drifting" if share >= 0.1 else "stable")

    result = {
        "verdict": verdict,
        "n_features_drifted": n_drift,
        "n_features_total": len(n_feats),
        "prediction_ks": round(pred_ks, 4),
        "prediction_drifted": bool(pred_drift),
        "features": sorted(n_feats, key=lambda x: -x["ks"]),
        "advice": (
            "distributionvalid,has SHAP "
            if verdict == "stable"
            else "distributiondrift:backgrounddistribution,recommendupdatebackground(session.update_background)"
        ),
    }
    return result
