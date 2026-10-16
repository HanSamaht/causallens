"""Steering vector 修复闭环：诊断 → 治疗 → 复诊。

原理（RepE / activation steering，Zou et al. 2023）：
  从"可疑特征高值组 vs 低值组"的平均激活差中提取方向向量，
  推理时在该层加上/减去向量（不改权重），削弱模型对可疑特征的依赖。

MIC 的独特之处在第三步——复诊：用自家审讯工具（归因占比、忠实性、预测漂移）
量化修复效果，修不好就如实报告并降级为"建议重训"。
"""
from __future__ import annotations

import numpy as np

from ..core.session import MicSession
from ..drivers.torch_model import TorchDriver
from .attribution import shap_matrix


def _feature_share(session: MicSession, feature: str) -> float:
    bg = session.driver.background[:200]
    sm = np.abs(shap_matrix(session.driver, bg)).mean(axis=0)
    j = session.driver.feature_names.index(feature)
    return float(sm[j] / (sm.sum() or 1.0))


def _mean_activation(drv: TorchDriver, idxs: np.ndarray, layer: str) -> np.ndarray:
    acts = []
    for i in idxs:
        acts.append(drv.get_activations(drv.sample_at(int(i)))[layer])
    return np.mean(np.stack(acts), axis=0)


def steering_repair(
    session: MicSession,
    feature: str,
    layer: str | None = None,
    alphas: tuple[float, ...] = (0.5, 1.0, 1.5),
    n_group: int = 24,
) -> dict:
    """尝试用 steering vector 削弱模型对 `feature` 的依赖，并复诊验证。"""
    drv = session.driver
    assert drv.kind == "torch", "修复闭环当前支持 torch 模型"
    j = drv.feature_names.index(feature)

    # ---- 诊断（before） ----
    bg = drv.background
    col = bg[:, j]
    high_idx = np.argsort(col)[-n_group:]
    low_idx = np.argsort(col)[:n_group]
    share_before = _feature_share(session, feature)
    p_before = drv.predict(bg[:200])

    # ---- 提取方向向量：高值组 - 低值组 的平均激活差 ----
    layer = layer or drv.list_layers()[max(0, len(drv.list_layers()) // 2)]
    v_high = _mean_activation(drv, high_idx, layer)
    v_low = _mean_activation(drv, low_idx, layer)
    delta = v_high - v_low  # "可疑特征高值时激活长什么样"

    # ---- 治疗：扫描 (alpha, sign)，用复诊指标选最优 ----
    trials = []
    for sign in (+1.0, -1.0):
        for a in alphas:
            vec = -sign * a * delta  # 减弱依赖：默认往"低值方向"推
            drv._steer_plan[layer] = vec.astype(np.float32)
            try:
                share_after = _feature_share(session, feature)
                p_after = drv.predict(bg[:200])
                pred_shift = float(np.mean(np.abs(p_after - p_before)))
            finally:
                drv._steer_plan.pop(layer, None)
            trials.append(
                {
                    "layer": layer,
                    "sign": "+" if sign > 0 else "-",
                    "alpha": a,
                    "feature_share": round(share_after, 4),
                    "share_reduction": round(share_before - share_after, 4),
                    "mean_pred_shift": round(pred_shift, 4),
                }
            )

    # 最优 = 占比下降最多，且预测整体漂移可接受（<0.15）
    feasible = [t for t in trials if t["mean_pred_shift"] < 0.15] or trials
    best = max(feasible, key=lambda t: t["share_reduction"])

    # ---- 复诊（after，按最优参数重放） ----
    sign = 1.0 if best["sign"] == "+" else -1.0
    drv._steer_plan[layer] = (-sign * best["alpha"] * delta).astype(np.float32)
    try:
        share_final = _feature_share(session, feature)
        audit_after = session.audit(None)
    finally:
        drv._steer_plan.pop(layer, None)

    success = share_final < share_before * 0.8  # 占比降两成以上才算修复有效
    return {
        "diagnosis": {
            "feature": feature,
            "concept": session.pack.concept_of(feature.split("@")[0]),
            "attribution_share_before": round(share_before, 4),
        },
        "treatment": {
            "layer": layer,
            "vector": f"mean_act(high) - mean_act(low) @ {layer}",
            "trials": trials,
            "best": best,
        },
        "follow_up": {
            "attribution_share_after": round(share_final, 4),
            "audit_findings_after": [f["message"] for f in audit_after["findings"]],
        },
        "verdict": (
            "修复有效：steering 补丁可用（推理时施加，无需重训）"
            if success
            else "修复效果不足：建议回退到数据清洗/重训路线（移除或正则该特征后对比）"
        ),
        "note": "steering 仅改变推理行为，不改变模型权重；生产使用需对全量验证集复诊",
    }
