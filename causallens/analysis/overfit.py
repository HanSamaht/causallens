"""deflated(Deflated Sharpe Ratio, Bailey & López de Prado 2014)——.

problem:"thisbacktest,' N most',also?"
PBO/CSCV requireretrain(its,not);DSR is——isauditthehas.
"""
from __future__ import annotations

import math

_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam has( ~1e-9), scipy."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p mustin (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def sharpe_stats(returns) -> dict:
    """ -> SR(every)///n."""
    import numpy as np

    r = np.asarray(returns, dtype=float)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sr = mu / sd if sd > 1e-12 else 0.0
    z = (r - mu) / (sd + 1e-12)
    return {"n": int(len(r)), "sharpe_per_period": round(sr, 4),
            "skew": round(float(z.mean() ** 0 + (z ** 3).mean()), 4) if len(r) > 3 else None,
            "kurtosis": round(float((z ** 4).mean()), 4) if len(r) > 4 else None}


def deflated_sharpe(returns, n_trials: int) -> dict:
    """DSR:SR N ,P(SR>0) is.

    returns: (/bar );n_trials: asthis.
    """
    import numpy as np

    st = sharpe_stats(returns)
    sr, T = st["sharpe_per_period"], st["n"]
    g3 = st["skew"] or 0.0
    g4 = st["kurtosis"] or 3.0
    if T < 5 or n_trials < 1:
        return {"error": "sample <5 or n_trials <1"}
    N = max(n_trials, 2)

    # SR and"hypothesisbelowmostSR"(on)
    var_sr = max((1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (T - 1), 1e-12)
    e_max = math.sqrt(var_sr) * (
        (1 - _EULER_GAMMA) * _norm_ppf(1 - 1.0 / N) + _EULER_GAMMA * _norm_ppf(1 - 1.0 / (N * math.e))
    )
    dsr = _norm_cdf((sr - e_max) / math.sqrt(var_sr))
    return {
        **st,
        "n_trials": int(N),
        "sr_expected_max_by_luck": round(e_max, 4),
        "DSR": round(dsr, 4),
        "verdict": (
            f"deflated(DSR={dsr:.2f}): {N} SR>0 probabilityhas {dsr:.0%}"
            if dsr >= 0.95
            else f"deflated(DSR={dsr:.2f}): SR onis {e_max:.3f},"
                 f" {sr:.3f}——bias edge"
        ),
        "note": "Bailey & López de Prado (2014).:PBO/CSCV(retrain,frameworkonlynot)",
    }
