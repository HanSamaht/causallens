"""版本对比审讯（model diff）：两个模型版本的行为差异画像。

回答制作者最关心的问题："我改了数据/结构重训后，模型行为到底变了哪里？"
  - 预测层：分歧率（跨阈值翻转率）、预测分布偏移
  - 归因层：全局归因占比的逐特征差分（谁的权重升了/降了）
  - 个案层：分歧最大的样本（改了预测的典型案例，附两版归因对比）
"""
from __future__ import annotations

import numpy as np

from ..core.concepts import ConceptPack
from ..drivers.base import MicDriver
from .attribution import shap_matrix
from .counterfactual import default_threshold


def diff_sessions(
    driver_old: MicDriver,
    driver_new: MicDriver,
    pack: ConceptPack | None = None,
    max_cases: int = 3,
) -> dict:
    assert driver_old.feature_names == driver_new.feature_names, "两个版本特征空间必须一致"
    pack = pack or ConceptPack()
    n = min(len(driver_old.background), len(driver_new.background), 500)
    X = driver_old.background[:n]

    p_old = driver_old.predict(X)
    p_new = driver_new.predict(X)
    thr_o = default_threshold(driver_old)
    thr_n = default_threshold(driver_new)
    sign_old = (p_old > thr_o).astype(int) if driver_old.prediction_semantics == "probability" else np.sign(p_old)
    sign_new = (p_new > thr_n).astype(int) if driver_new.prediction_semantics == "probability" else np.sign(p_new)
    disagreement = sign_old != sign_new
    flip_rate = float(disagreement.mean())
    delta_pred = p_new - p_old

    # 全局归因占比差分
    sm_o = np.abs(shap_matrix(driver_old, X)).mean(axis=0)
    sm_n = np.abs(shap_matrix(driver_new, X)).mean(axis=0)
    share_o, share_n = sm_o / (sm_o.sum() or 1), sm_n / (sm_n.sum() or 1)
    d_share = share_n - share_o

    feature_diffs = sorted(
        (
            {
                "feature": f,
                "concept": pack.concept_of(f.split("@")[0]),
                "share_old": round(float(share_o[j]), 4),
                "share_new": round(float(share_n[j]), 4),
                "delta": round(float(d_share[j]), 4),
            }
            for j, f in enumerate(driver_old.feature_names)
        ),
        key=lambda x: -abs(x["delta"]),
    )

    # 分歧最大的典型案例
    order = np.argsort(-np.abs(delta_pred) * disagreement)
    cases = []
    for i in order[:max_cases]:
        if not disagreement[i]:
            break
        cases.append(
            {
                "sample_index": int(i),
                "prediction_old": round(float(p_old[i]), 4),
                "prediction_new": round(float(p_new[i]), 4),
                "top_attribution_old": driver_old.feature_names[int(np.argmax(-np.abs(shap_matrix(driver_old, X[i : i + 1]))[0]))],
                "top_attribution_new": driver_new.feature_names[int(np.argmax(-np.abs(shap_matrix(driver_new, X[i : i + 1]))[0]))],
            }
        )

    return {
        "n_compared": int(n),
        "agreement_rate": round(1 - flip_rate, 4),
        "flip_rate": round(flip_rate, 4),
        "mean_prediction_shift": round(float(np.mean(delta_pred)), 4),
        "max_abs_shift": round(float(np.max(np.abs(delta_pred))), 4),
        "attribution_diff": feature_diffs,
        "disagreement_cases": cases,
        "verdict": (
            "行为稳定"
            if flip_rate < 0.05 and all(abs(d["delta"]) < 0.1 for d in feature_diffs[:3])
            else "行为有实质变化——见 attribution_diff 与 disagreement_cases"
        ),
    }
