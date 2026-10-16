"""漂移检测（零依赖：KS 统计量 + PSI + 预测分布漂移）。

部署后分布漂移会让 SHAP 基准悄悄失效——本模块回答"背景参考分布还作数吗"。
不引入 evidently/alibi-detect 重型依赖（它们可选用于生成 HTML 报告），
KS/PSI 自实现，确定性与可测试优先。

  - KS：两样本 Kolmogorov-Smirnov 统计量，α=0.05 临界值 1.36*sqrt((n+m)/(nm))
  - PSI：Population Stability Index，>0.25 显著漂移（业界惯例）
  - 预测漂移：模型输出分布在参考/新数据上的 KS
"""
from __future__ import annotations

import numpy as np

from ..drivers.base import MicDriver


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """两样本 KS 统计量 D（零依赖实现）。"""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    all_v = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, all_v, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_v, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _ks_pvalue_approx(d: float, n: int, m: int) -> float:
    """渐近 p 值（Smirnov 近似），用于展示而非严格推断。"""
    ne = n * m / (n + m)
    lam = (np.sqrt(ne) + 0.12 + 0.11 / np.sqrt(ne)) * d
    p = 2.0 * sum((-1) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam) for k in range(1, 101))
    return float(max(0.0, min(1.0, p)))


def _psi(a: np.ndarray, b: np.ndarray, bins: int = 10) -> float:
    """PSI：以参考分布分位分箱（分位箱保证每组有样本）。"""
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
    """参考背景 vs 新数据的逐特征漂移 + 预测分布漂移。

    返回 overall_verdict: stable / drifting / severely_drifting。
    漂移被检出时，应提醒"当前 SHAP 基准基于旧参考分布，解释结论可能失效"。
    """
    ref = driver.background
    new = np.asarray(new_data, dtype=float)
    assert new.shape[1] == ref.shape[1], "新数据列数与背景不一致"

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
            "参考分布仍然有效，现有 SHAP 基准可信"
            if verdict == "stable"
            else "检出分布漂移：背景参考分布已过时，建议用新数据更新背景（session.update_background）后再解释"
        ),
    }
    return result
