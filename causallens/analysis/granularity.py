"""attribution:channel / segment / full / gradient + auto .

:
  - TimeSHAP (KDD 2021, 274 ):layerattribution(cell->timestep->feature->global)
  - GS-SHAP (arXiv 2601.06114, 2026): 1.7×
  - FastSHAP (ICLR 2022, 337 ): SHAP,result
"""

from __future__ import annotations

import numpy as np

from .attribution import shap_matrix


def _flat_to_matrix(flat: np.ndarray, T: int, F: int) -> np.ndarray:
    return flat.reshape(T, F)


def _aggregate_channel(shap_flat: np.ndarray, T: int, F: int, base_names: list[str]) -> list[dict]:
    """:everyinallon SHAP and."""
    m = _flat_to_matrix(shap_flat, T, F) # (T, F)
    vals = m.sum(axis=0) # (F,)
    total = np.abs(vals).sum() or 1.0
    return [
        {"feature": base_names[j], "concept": base_names[j], "shap": float(vals[j]),
         "share": float(abs(vals[j]) / total), "value": float(vals[j])}
        for j in range(F)
    ]


def _aggregate_segment(shap_flat: np.ndarray, T: int, F: int, base_names: list[str],
                       n_segments: int = 4) -> list[dict]:
    """×:everyineveryon SHAP and."""
    m = _flat_to_matrix(shap_flat, T, F)
    seg_size = max(1, T // n_segments)
    total = np.abs(m).sum() or 1.0
    rows = []
    for j in range(F):
        for s in range(n_segments):
            lo, hi = s * seg_size, min((s + 1) * seg_size, T)
            v = float(m[lo:hi, j].sum())
            seg_label = "" if s == 0 else ("" if s == n_segments - 1 else f"{s+1}")
            rows.append({
                "feature": f"{base_names[j]}@{seg_label}",
                "concept": f"{base_names[j]}·{seg_label}",
                "shap": v, "share": float(abs(v) / total), "value": v,
            })
    return rows


def _aggregate_full(shap_flat: np.ndarray, T: int, F: int, base_names: list[str]) -> list[dict]:
    """:everyinevery lag on SHAP."""
    total = np.abs(shap_flat).sum() or 1.0
    return [
        {"feature": f"{base_names[j]}@lag{T-1-t}", "concept": f"{base_names[j]}@lag{T-1-t}",
         "shap": float(shap_flat[t * F + j]),
         "share": float(abs(shap_flat[t * F + j]) / total),
         "value": float(shap_flat[t * F + j])}
        for t in range(T) for j in range(F)
    ]


def _gradient_attribution(driver, x: np.ndarray, T: int, F: int, base_names: list[str]) -> list[dict]:
    """×input(,)."""
    if hasattr(driver, "input_attribution"):
        g = driver.input_attribution(x)
    elif hasattr(driver, "window_attribution"):
        heat, _ = driver.window_attribution(x)
        g = heat.reshape(-1)
    else:
        raise ValueError("driverattributionmethod")
    total = np.abs(g).sum() or 1.0
    # to
    m = _flat_to_matrix(g, T, F)
    vals = m.sum(axis=0)
    return [
        {"feature": base_names[j], "concept": base_names[j], "shap": float(vals[j]),
         "share": float(abs(vals[j]) / total), "value": float(vals[j])}
        for j in range(F)
    ]


def recommend_granularity(n_flat_features: int) -> str:
    """auto :featuremost."""
    if n_flat_features <= 100:
        return "full"
    elif n_flat_features <= 500:
        return "segment"
    else:
        return "channel"


def timeseries_attribution(driver, x: np.ndarray, granularity: str = "auto",
                           n_segments: int = 4, top_k: int = 10) -> dict:
    """modelattribution.

    granularity:
      "channel" — (F feature,most,most)
      "segment" — ×(F×n_segments,inetc,mode)
      "full" — (T×F feature,most,)
      "gradient" — (,)
      "auto" —
    """
    base_names = getattr(driver, "base_feature_names", driver.feature_names)
    T = getattr(driver, "seq_len", 1)
    F = getattr(driver, "n_features_raw", len(base_names))
    n_flat = T * F

    if granularity == "auto":
        granularity = recommend_granularity(n_flat)

    x_flat = np.asarray(x, dtype=float).ravel()

    if granularity == "gradient":
        items = _gradient_attribution(driver, x_flat, T, F, base_names)
        engine = "gradient×input->channel"
    elif granularity == "channel":
        sv = shap_matrix(driver, x_flat.reshape(1, -1))[0]
        # SHAP (etcinput SHAP)
        items = _aggregate_channel(sv, T, F, base_names)
        engine = f"SHAP->({F}feature)"
    elif granularity == "segment":
        sv = shap_matrix(driver, x_flat.reshape(1, -1))[0]
        items = _aggregate_segment(sv, T, F, base_names, n_segments)
        engine = f"SHAP->×{n_segments}({F * n_segments}feature)"
    elif granularity == "full":
        sv = shap_matrix(driver, x_flat.reshape(1, -1))[0]
        items = _aggregate_full(sv, T, F, base_names)
        engine = f"SHAP({n_flat}feature)"
    else:
        raise ValueError(f": {granularity}")

    items.sort(key=lambda a: abs(a["shap"]), reverse=True)
    p = driver.predict_one(x_flat)
    return {
        "granularity": granularity,
        "engine": engine,
        "n_input_features_raw": n_flat,
        "n_output_features": len(items),
        "prediction": round(float(p), 4),
        "attributions": [
            {**a, "role": "normal"} for a in items[:top_k]
        ],
    }
