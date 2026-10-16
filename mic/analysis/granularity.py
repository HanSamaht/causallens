"""四档时序归因：channel / segment / full / gradient + auto 推荐。

文献依据：
  - TimeSHAP (KDD 2021, 274 引)：多层级归因（cell→timestep→feature→global）
  - GS-SHAP (arXiv 2601.06114, 2026)：分组分段后忠实度提升 1.7×
  - FastSHAP (ICLR 2022, 337 引)：摊销式 SHAP，一次前向出全分辨率结果
"""

from __future__ import annotations

import numpy as np

from .attribution import shap_matrix


def _flat_to_matrix(flat: np.ndarray, T: int, F: int) -> np.ndarray:
    return flat.reshape(T, F)


def _aggregate_channel(shap_flat: np.ndarray, T: int, F: int, base_names: list[str]) -> list[dict]:
    """通道级聚合：每个通道在全部时间步上的 SHAP 求和。"""
    m = _flat_to_matrix(shap_flat, T, F)  # (T, F)
    vals = m.sum(axis=0)  # (F,)
    total = np.abs(vals).sum() or 1.0
    return [
        {"feature": base_names[j], "concept": base_names[j], "shap": float(vals[j]),
         "share": float(abs(vals[j]) / total), "value": float(vals[j])}
        for j in range(F)
    ]


def _aggregate_segment(shap_flat: np.ndarray, T: int, F: int, base_names: list[str],
                       n_segments: int = 4) -> list[dict]:
    """通道×时段聚合：每个通道在每个时间分段上的 SHAP 求和。"""
    m = _flat_to_matrix(shap_flat, T, F)
    seg_size = max(1, T // n_segments)
    total = np.abs(m).sum() or 1.0
    rows = []
    for j in range(F):
        for s in range(n_segments):
            lo, hi = s * seg_size, min((s + 1) * seg_size, T)
            v = float(m[lo:hi, j].sum())
            seg_label = "前段" if s == 0 else ("后段" if s == n_segments - 1 else f"段{s+1}")
            rows.append({
                "feature": f"{base_names[j]}@{seg_label}",
                "concept": f"{base_names[j]}·{seg_label}",
                "shap": v, "share": float(abs(v) / total), "value": v,
            })
    return rows


def _aggregate_full(shap_flat: np.ndarray, T: int, F: int, base_names: list[str]) -> list[dict]:
    """全分辨率：每个通道在每个 lag 上的 SHAP。"""
    total = np.abs(shap_flat).sum() or 1.0
    return [
        {"feature": f"{base_names[j]}@lag{T-1-t}", "concept": f"{base_names[j]}@lag{T-1-t}",
         "shap": float(shap_flat[t * F + j]),
         "share": float(abs(shap_flat[t * F + j]) / total),
         "value": float(shap_flat[t * F + j])}
        for t in range(T) for j in range(F)
    ]


def _gradient_attribution(driver, x: np.ndarray, T: int, F: int, base_names: list[str]) -> list[dict]:
    """梯度×输入兜底（毫秒级，近似）。"""
    if hasattr(driver, "input_attribution"):
        g = driver.input_attribution(x)
    elif hasattr(driver, "window_attribution"):
        heat, _ = driver.window_attribution(x)
        g = heat.reshape(-1)
    else:
        raise ValueError("驱动器无梯度归因方法")
    total = np.abs(g).sum() or 1.0
    # 聚合到通道级
    m = _flat_to_matrix(g, T, F)
    vals = m.sum(axis=0)
    return [
        {"feature": base_names[j], "concept": base_names[j], "shap": float(vals[j]),
         "share": float(abs(vals[j]) / total), "value": float(vals[j])}
        for j in range(F)
    ]


def recommend_granularity(n_flat_features: int) -> str:
    """auto 推荐逻辑：根据展平特征数选最佳档位。"""
    if n_flat_features <= 100:
        return "full"
    elif n_flat_features <= 500:
        return "segment"
    else:
        return "channel"


def timeseries_attribution(driver, x: np.ndarray, granularity: str = "auto",
                           n_segments: int = 4, top_k: int = 10) -> dict:
    """时序模型的多档归因入口。

    granularity:
      "channel"  — 通道级（F 个特征，最快，最可读）
      "segment"  — 通道×时段（F×n_segments，中等，保留时间模式）
      "full"     — 全分辨率（T×F 个特征，最精确，可能慢）
      "gradient" — 梯度近似（通道级聚合，毫秒级）
      "auto"     — 自动推荐
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
        engine = "gradient×input→channel聚合"
    elif granularity == "channel":
        sv = shap_matrix(driver, x_flat.reshape(1, -1))[0]
        # 对展平 SHAP 做通道聚合（等价于对输入分组后算 SHAP）
        items = _aggregate_channel(sv, T, F, base_names)
        engine = f"SHAP→通道聚合({F}特征)"
    elif granularity == "segment":
        sv = shap_matrix(driver, x_flat.reshape(1, -1))[0]
        items = _aggregate_segment(sv, T, F, base_names, n_segments)
        engine = f"SHAP→通道×{n_segments}段({F * n_segments}特征)"
    elif granularity == "full":
        sv = shap_matrix(driver, x_flat.reshape(1, -1))[0]
        items = _aggregate_full(sv, T, F, base_names)
        engine = f"SHAP全分辨率({n_flat}特征)"
    else:
        raise ValueError(f"未知粒度: {granularity}")

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
