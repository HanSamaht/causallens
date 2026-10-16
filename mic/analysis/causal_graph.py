"""干预式因果图（lite）。

哲学约束：区分两类证据——
  - direct 边（特征→预测）：SHAP 聚合的总效应（近似干预效应，树模型下较可靠）
  - interaction 边（特征 f→ 特征 g）：真正的干预实验——把 f do() 到不同分位值，
    观察 g 的归因如何变化。这不是相关，是干预。
  - correlational 边：仅背景数据上的观测相关，供审查员对照"哪些边可能只是相关"。

MVP 的图是"特征层 → 预测"的浅层因果图；激活层因果图（circuit tracing）属于
TorchDriver + patch_activation 的进阶路径。
"""
from __future__ import annotations

import numpy as np

from ..core.types import CausalEdge, CausalGraph
from ..drivers.base import MicDriver
from .attribution import shap_matrix


def build_causal_graph(
    driver: MicDriver,
    anchor: np.ndarray,
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
    max_interaction_edges: int = 8,
    interaction_threshold: float | None = None,
    correlational: bool = True,
) -> CausalGraph:
    anchor = np.asarray(anchor, dtype=float).ravel()
    feats = driver.feature_names

    # ---- 1) direct 边：背景样本上平均 |SHAP| ----
    bg = driver.background[: min(len(driver.background), 200)]
    sm = shap_matrix(driver, bg)
    mean_abs = np.abs(sm).mean(axis=0)
    denom = mean_abs.sum() or 1.0

    edges = [
        CausalEdge(
            source=f,
            target="prediction",
            effect=float(mean_abs[j]),
            kind="direct",
            detail=f"归因占比 {mean_abs[j] / denom:.1%}",
        )
        for j, f in enumerate(feats)
    ]

    # ---- 2) interaction 边：干预 f，观察 g 的归因漂移 ----
    base_shap = shap_matrix(driver, anchor.reshape(1, -1))[0]
    drift = np.zeros((len(feats), len(feats)))  # drift[f_idx, g_idx]
    for j, f in enumerate(feats):
        for q in quantiles:
            x2 = driver.intervene(anchor, {f: driver.quantile_of(f, q)})
            s2 = shap_matrix(driver, x2.reshape(1, -1))[0]
            drift[j] += np.abs(s2 - base_shap) / len(quantiles)
    np.fill_diagonal(drift, 0.0)

    inter: list[tuple[float, int, int]] = []
    for jf in range(len(feats)):
        for jg in range(len(feats)):
            if jf != jg and drift[jf, jg] > 0:
                inter.append((drift[jf, jg], jf, jg))
    inter.sort(reverse=True)
    thr = interaction_threshold if interaction_threshold is not None else (
        inter[0][0] * 0.15 if inter else 0.0
    )
    for eff, jf, jg in inter[:max_interaction_edges]:
        if eff < thr:
            break
        edges.append(
            CausalEdge(
                source=feats[jf],
                target=feats[jg],
                effect=float(eff),
                kind="interaction",
                detail="干预 f 后 g 的归因平均漂移（干预证据）",
            )
        )

    # ---- 3) correlational 边：仅供对照 ----
    if correlational:
        corr = np.corrcoef(driver.background, rowvar=False)
        n_feats = len(feats)
        pairs = sorted(
            ((abs(corr[i, j]), i, j) for i in range(n_feats) for j in range(i + 1, n_feats)),
            reverse=True,
        )[:max_interaction_edges]
        for c, i, j in pairs:
            if c > 0.5:
                edges.append(
                    CausalEdge(
                        source=feats[i],
                        target=feats[j],
                        effect=float(corr[i, j]),
                        kind="correlational",
                        detail="背景数据观测相关（非因果证据，供审查对照）",
                    )
                )

    return CausalGraph(nodes=feats + ["prediction"], edges=edges)
