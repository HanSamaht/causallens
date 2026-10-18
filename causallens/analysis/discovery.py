"""Structure discovery backends from the causal-learn library.

The intervention graph in causal_graph.py answers "how strong is this
edge". This module answers a different question: "which edges should
exist at all". It runs constraint-based or functional discovery on the
background data augmented with the model prediction column, so the
prediction node is a first-class citizen of the discovered structure.

Supported methods:
  - "pc":     PC constraint-based algorithm (conditional independence)
  - "ges":    Greedy Equivalence Search (BIC score)
  - "lingam": DirectLiNGAM (linear non-Gaussian, gives a full DAG)

All edges returned carry kind "discovered". Edges into the prediction
node are hypothesis generators: they must survive the intervention
experiments in causal_graph.py before they are trusted.
"""
from __future__ import annotations

import numpy as np

from ..core.types import CausalEdge

try:  # optional dependency, soft fallback
    from causallearn.search.ConstraintBased.PC import pc as _pc
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    from causallearn.search.ScoreBased.GES import ges as _ges

    HAS_CAUSAL_LEARN = True
except Exception:  # pragma: no cover - exercised only without the library
    HAS_CAUSAL_LEARN = False

DISCOVERY_METHODS = ("pc", "ges", "lingam")


def _augment_with_predictions(driver, data: np.ndarray) -> np.ndarray:
    """Append the model prediction column to the background matrix."""
    preds = driver.predict(data).reshape(-1, 1)
    return np.hstack([np.asarray(data, dtype=float), preds])


def _cpdag_edges(graph: np.ndarray, names: list[str]):
    """Extract edges from a causal-learn CPDAG matrix.

    Encoding: graph[i, j] == -1 and graph[j, i] == 1 means i -> j.
    graph[i, j] == -1 and graph[j, i] == -1 is an undirected edge whose
    orientation the algorithm cannot determine from data alone.

    Returns (i, j, effect, directed) tuples.
    """
    n = len(names)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if graph[i, j] == -1 and graph[j, i] == 1:
                out.append((i, j, 1.0, True))
            elif graph[i, j] == 1 and graph[j, i] == -1:
                out.append((j, i, 1.0, True))
            elif graph[i, j] == -1 and graph[j, i] == -1:
                out.append((i, j, 1.0, False))
    return out


def _edges_from_adjacency(adj: np.ndarray, names: list[str], weight_cap: int):
    """Directed edges from a weighted adjacency matrix (LiNGAM style)."""
    out = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if i != j and abs(adj[i, j]) > 1e-6:
                out.append((i, j, float(adj[i, j])))
    out.sort(key=lambda t: abs(t[2]), reverse=True)
    return out[:weight_cap]


def discovery_edges(
    driver,
    method: str = "pc",
    alpha: float = 0.05,
    max_samples: int = 400,
    max_edges: int = 15,
) -> list[CausalEdge]:
    """Run structure discovery over features plus the prediction node.

    Returns edges of kind "discovered". Feature-to-feature edges are
    structural hypotheses. Feature-to-prediction edges mark features the
    discovery algorithm places as direct causes of the model output.
    Prediction-to-feature edges, if any, are diagnostics: the discovery
    backend sees the output as upstream of an input, which usually means
    the feature is a deterministic function of other inputs.
    """
    if not HAS_CAUSAL_LEARN:
        raise ImportError(
            "causal-learn is not installed. Install it with: pip install causal-learn"
        )
    if method not in DISCOVERY_METHODS:
        raise ValueError(f"unknown discovery method {method!r}, expected one of {DISCOVERY_METHODS}")

    bg = driver.background[:max_samples]
    data = _augment_with_predictions(driver, bg)
    names = list(driver.feature_names) + ["prediction"]
    target_idx = len(names) - 1

    directed: list[tuple[int, int, float, bool]] = []

    if method == "pc":
        res = _pc(data, alpha=alpha, indep_test="fisherz", show_progress=False)
        directed = _cpdag_edges(res.G.graph, names)
    elif method == "ges":
        res = _ges(data, score_func="local_score_BIC")
        directed = _cpdag_edges(res["G"].graph, names)
    else:  # lingam
        model = DirectLiNGAM()
        model.fit(data)
        weighted = _edges_from_adjacency(model.adjacency_matrix_, names, max_edges)
        directed = [(i, j, w, True) for i, j, w in weighted]

    # Prediction-related edges always survive the cap; the rest compete.
    pred_edges = [e for e in directed if target_idx in (e[0], e[1])]
    struct_edges = [e for e in directed if target_idx not in (e[0], e[1])]
    keep = pred_edges + struct_edges[: max(0, max_edges - len(pred_edges))]

    edges: list[CausalEdge] = []
    for i, j, w, is_directed in keep:
        src, dst = names[i], names[j]
        arrow = "->" if is_directed else "-"
        if j == target_idx:
            detail = f"{method} discovery: {src} {arrow} prediction direct cause"
        elif i == target_idx:
            detail = (
                f"{method} discovery: prediction placed upstream of {dst}; "
                "the feature is likely a deterministic function of other inputs"
            )
        else:
            note = "" if is_directed else " (orientation undetermined, CPDAG)"
            detail = f"{method} discovery: {src} {arrow} {dst} structural hypothesis{note}"
        edges.append(CausalEdge(source=src, target=dst, effect=float(w), kind="discovered", detail=detail))
    return edges
