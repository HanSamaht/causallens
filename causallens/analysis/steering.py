"""Steering vector repair:diagnosis -> treatment -> follow-up.

(RepE / activation steering,Zou et al. 2023):
  from"suspiciousfeaturevalue vs value"activationindirection,
  inthelayeron/(notweight),modelsuspiciousfeature.

MIC in——follow-up:interrogationtool(attribution,,predictiondrift)
repair,notreportas"recommendretrain".
"""
from __future__ import annotations

import numpy as np

from ..core.session import MicSession
from ..drivers.torch_model import TorchDriver
from .attribution import shap_matrix


def _feature_share(session: MicSession, feature: str) -> float:
    """Feature share."""
    bg = session.driver.background[:200]
    sm = np.abs(shap_matrix(session.driver, bg)).mean(axis=0)
    j = session.driver.feature_names.index(feature)
    return float(sm[j] / (sm.sum() or 1.0))


def _mean_activation(drv: TorchDriver, idxs: np.ndarray, layer: str) -> np.ndarray:
    """Mean activation."""
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
    """ steering vector model `feature` ,follow-upvalidate."""
    drv = session.driver
    assert drv.kind == "torch", "repairwhensupport torch model"
    j = drv.feature_names.index(feature)

    # ---- diagnosis(before) ----
    bg = drv.background
    col = bg[:, j]
    high_idx = np.argsort(col)[-n_group:]
    low_idx = np.argsort(col)[:n_group]
    share_before = _feature_share(session, feature)
    p_before = drv.predict(bg[:200])

    # ---- direction:value - value activation ----
    layer = layer or drv.list_layers()[max(0, len(drv.list_layers()) // 2)]
    v_high = _mean_activation(drv, high_idx, layer)
    v_low = _mean_activation(drv, low_idx, layer)
    delta = v_high - v_low # "suspiciousfeaturevalueactivation"

    # ---- treatment:scan (alpha, sign),follow-upmost ----
    trials = []
    for sign in (+1.0, -1.0):
        for a in alphas:
            vec = -sign * a * delta # :default"valuedirection"
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

    # most = belowmost,predictiondriftaccept(<0.15)
    feasible = [t for t in trials if t["mean_pred_shift"] < 0.15] or trials
    best = max(feasible, key=lambda t: t["share_reduction"])

    # ---- follow-up(after,mostparameter) ----
    sign = 1.0 if best["sign"] == "+" else -1.0
    drv._steer_plan[layer] = (-sign * best["alpha"] * delta).astype(np.float32)
    try:
        share_final = _feature_share(session, feature)
        audit_after = session.audit(None)
    finally:
        drv._steer_plan.pop(layer, None)

    success = share_final < share_before * 0.8 # byonrepairvalid
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
            "repairvalid:steering (,retrain)"
            if success
            else "repairinsufficient:recommendfallbackto/retrain(removeorthefeaturecontrast)"
        ),
        "note": "steering behavior,notmodelweight;validatefollow-up",
    }
