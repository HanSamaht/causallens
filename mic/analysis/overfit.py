"""缩水夏普（Deflated Sharpe Ratio, Bailey & López de Prado 2014）——零依赖实现。

回答的问题："这个回测数字，扣掉'试了 N 个策略挑最好'的运气后，还剩多少可信度？"
PBO/CSCV 需要重训循环（引用其论文、不集成）；DSR 是纯公式——正是审计该有的形态。
"""
from __future__ import annotations

import math

_EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam 有理逼近（精度 ~1e-9），免 scipy。"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p 必须在 (0,1)")
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
    """收益序列 → SR（每期）/偏度/峰度/n。"""
    import numpy as np

    r = np.asarray(returns, dtype=float)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sr = mu / sd if sd > 1e-12 else 0.0
    z = (r - mu) / (sd + 1e-12)
    return {"n": int(len(r)), "sharpe_per_period": round(sr, 4),
            "skew": round(float(z.mean() ** 0 + (z ** 3).mean()), 4) if len(r) > 3 else None,
            "kurtosis": round(float((z ** 4).mean()), 4) if len(r) > 4 else None}


def deflated_sharpe(returns, n_trials: int) -> dict:
    """DSR：SR 扣掉 N 次策略挑选的运气后，P(真实SR>0) 是多少。

    returns: 单期收益序列（日/bar 级）；n_trials: 为了选出这个策略试了多少个。
    """
    import numpy as np

    st = sharpe_stats(returns)
    sr, T = st["sharpe_per_period"], st["n"]
    g3 = st["skew"] or 0.0
    g4 = st["kurtosis"] or 3.0
    if T < 5 or n_trials < 1:
        return {"error": "样本 <5 或 n_trials <1"}
    N = max(n_trials, 2)

    # SR 估计方差与"零假设下的最大期望SR"（多重检验上限）
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
            f"扛得住缩水（DSR={dsr:.2f}）：扣掉 {N} 次挑选运气后 SR>0 的概率仍有 {dsr:.0%}"
            if dsr >= 0.95
            else f"缩水后存疑（DSR={dsr:.2f}）：运气就能解释的 SR 上限是 {e_max:.3f}，"
                 f"实测 {sr:.3f}——先怀疑挑选偏差再相信 edge"
        ),
        "note": "Bailey & López de Prado (2014)。配套读法：PBO/CSCV（需重训循环，本框架只引用不集成）",
    }
