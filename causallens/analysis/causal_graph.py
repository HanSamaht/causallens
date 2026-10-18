"""interventioncausal graph(lite).

:evidence——
  - direct (feature->prediction):SHAP effect(interventioneffect,modelbelow)
  - interaction (feature f-> feature g):intervention experiment—— f do() tonotpercentilevalue,
     g attribution.thisnotcorrelation,isintervention.
  - correlational :backgroundoncorrelation,auditor"onlyiscorrelation".

MVP is"featurelayer -> prediction"layercausal graph;activationlayercausal graph(circuit tracing)
TorchDriver + patch_activation .
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
    discovery: str | None = None,
    discovery_alpha: float = 0.05,
) -> CausalGraph:
    """Build causal graph."""
    anchor = np.asarray(anchor, dtype=float).ravel()
    feats = driver.feature_names

    # ---- 0) structure discovery backend (causal-learn) ----
    discovered: list = []
    if discovery is not None:
        from .discovery import discovery_edges

        discovered = discovery_edges(driver, method=discovery, alpha=discovery_alpha)

    # ---- 1) direct :backgroundsampleon |SHAP| ----
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
            detail=f"attribution {mean_abs[j] / denom:.1%}",
        )
        for j, f in enumerate(feats)
    ]

    # ---- 2) interaction :intervention f, g attributiondrift ----
    base_shap = shap_matrix(driver, anchor.reshape(1, -1))[0]
    drift = np.zeros((len(feats), len(feats))) # drift[f_idx, g_idx]
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
                detail="intervention f g attributiondrift(interventionevidence)",
            )
        )

    # ---- 3) correlational : ----
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
                        detail="backgroundcorrelation(causalevidence,)",
                    )
                )

    edges.extend(discovered)
    method = "shap_aggregation + quantile_intervention"
    if discovery is not None:
        method += f" + {discovery}_discovery"
    return CausalGraph(nodes=feats + ["prediction"], edges=edges, method=method)
