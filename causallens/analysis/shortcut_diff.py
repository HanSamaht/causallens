"""Shortcut diff: model-implied reliance versus the discovered world graph.

A model can be right for the wrong reasons. The world graph (structure
discovered from the data, or supplied from domain knowledge) says which
features have a causal path to the outcome. The model-implied reliance
(measured by intervention contrasts on the frozen model) says which
features the model actually uses. The diff between the two is an audit:
reliance sitting on a feature with no causal path to the outcome is
reliance that breaks when the correlation that carries it breaks.

Each relied upon feature is classified:

  - CAUSAL PATH: a directed path to the outcome exists in the world
    graph; the reliance is structurally supported
  - PROXY: the outcome has a directed path into the feature; the model
    reads a consequence of the label, which is leakage or a legitimate
    pre-outcome measurement depending on timing the graph cannot see
  - UNRESOLVED PATH: connectivity exists but only through edges the
    discovery backend left undirected; the data cannot license the
    direction
  - INDEPENDENT: no connection at all in the discovered structure; the
    model leans on something the data says is unrelated to the outcome

The audit reports the share of interventional reliance at risk, the
per feature classification, and the caveats: CPDAG orientations are
assumptions, and PROXY needs deployment timing information.

Provenance: shortcut features characterized in DAG terms by Bellamy et
al. (2022), shortcut discovery in black box classifiers by Sahidullah
et al. (2025), the general failure mode by Geirhos et al. (2020). The
contribution here is the two graph diff as a single audit call with
reliance share accounting.
"""
from __future__ import annotations

from collections import deque

import numpy as np

try:  # optional dependency, same soft fallback as discovery.py
    from causallearn.search.ConstraintBased.PC import pc as _pc
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    from causallearn.search.ScoreBased.GES import ges as _ges

    from .discovery import _cpdag_edges

    HAS_CAUSAL_LEARN = True
except Exception:  # pragma: no cover - exercised only without the library
    HAS_CAUSAL_LEARN = False

_OUTCOME = "outcome"


def _directed_path(adj: dict[int, list[int]], src: int, dst: int) -> bool:
    seen = {src}
    queue = deque([src])
    while queue:
        node = queue.popleft()
        if node == dst:
            return True
        for nxt in adj.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def _world_edges(data: np.ndarray, names: list[str], method: str,
                 alpha: float) -> list[tuple[int, int, bool]]:
    """Edges of the world graph over features plus the outcome column.

    Returns (i, j, directed) tuples; directed is False for CPDAG edges
    whose orientation the backend could not determine.
    """
    if method == "pc":
        res = _pc(data, alpha=alpha, indep_test="fisherz",
                  max_cond_size=3, show_progress=False)
        return [(i, j, d) for i, j, _w, d in _cpdag_edges(res.G.graph, names)]
    if method == "ges":
        res = _ges(data, score_func="local_score_BIC")
        return [(i, j, d) for i, j, _w, d in _cpdag_edges(res["G"].graph, names)]
    if method == "lingam":
        model = DirectLiNGAM()
        model.fit(data)
        out = []
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i != j and abs(model.adjacency_matrix_[i, j]) > 1e-6:
                    out.append((i, j, True))
        return out
    raise ValueError(f"unknown discovery method {method!r}, expected pc, ges, or lingam")


def shortcut_diff(
    driver,
    y,
    method: str = "pc",
    alpha: float = 0.05,
    max_samples: int = 400,
    edges: list[tuple[str, str]] | None = None,
    min_reliance: float = 0.03,
    n_perm: int = 20,
    quantiles: tuple[float, float] = (0.10, 0.90),
    n_anchor: int = 64,
    seed: int = 0,
) -> dict:
    """Diff the model's interventional reliance against the world graph.

    Args:
        driver: any CausalLens driver (the model is only queried
            through predict).
        y: the real outcome aligned with driver.background. Discovery
            runs on the background plus this column; NaN inputs must be
            imputed by the caller first.
        method: structure discovery backend, "pc", "ges", or "lingam".
        alpha: conditional independence level for PC.
        max_samples: cap on rows for discovery, runtime guard.
        edges: domain knowledge edges (source, target) merged into the
            world graph as directed; they override an undirected pair.
        min_reliance: features below this interventional share are
            reported but not classified.
        n_perm: permutation replicates for the chance baseline of each
            reliance share (refit on shuffled outcome); 0 disables.
            Skipped automatically when the model family cannot be
            refit.
        quantiles: contrast pair defining each intervention.
        n_anchor: rows over which reliance is averaged.
        seed: reproducibility.

    Returns:
        Dict with per feature reliance and classification, the world
        edges, the reliance at risk, and one of two verdicts:
        SUPPORTED or SHORTCUT RISK.
    """
    if not HAS_CAUSAL_LEARN:
        raise ImportError(
            "causal-learn is not installed. Install it with: pip install causal-learn"
        )
    feats = list(driver.feature_names)
    bg = np.asarray(driver.background, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if len(y) != len(bg):
        raise ValueError(
            f"y has {len(y)} rows but the background has {len(bg)}; pass the "
            "training outcome aligned with driver.background"
        )
    if np.isnan(y).any() or np.isnan(bg).any():
        raise ValueError(
            "NaN in background or y; impute before calling shortcut_diff, "
            "the discovery backends cannot handle missing values"
        )
    if _OUTCOME in feats:
        raise ValueError(f"a feature is already named {_OUTCOME!r}, reserved")

    names = feats + [_OUTCOME]
    outcome_idx = len(names) - 1
    data = np.hstack([bg[:max_samples], y[:max_samples].reshape(-1, 1)])

    # constant columns make the correlation matrix singular
    stds = data.std(axis=0)
    keep = np.where(stds > 1e-12)[0].tolist()
    if len(keep) < 3:
        raise ValueError("fewer than 3 non-constant columns, nothing to discover")
    dropped = [names[j] for j in range(len(names)) if j not in keep]
    if dropped:
        data = data[:, keep]
        remap = {old: new for new, old in enumerate(keep)}
        names_kept = [names[j] for j in keep]
    else:
        remap = {i: i for i in range(len(names))}
        names_kept = names
    outcome_idx = remap[outcome_idx]

    world = _world_edges(data, names_kept, method, alpha)

    if edges:
        by_pair: dict[tuple[int, int], tuple[int, int, bool]] = {}
        for i, j, d in world:
            by_pair[(min(i, j), max(i, j))] = (i, j, d)
        for src, dst in edges:
            if src in names_kept and dst in names_kept:
                i, j = names_kept.index(src), names_kept.index(dst)
                by_pair[(min(i, j), max(i, j))] = (i, j, True)
        world = list(by_pair.values())

    directed: dict[int, list[int]] = {}
    skeleton: dict[int, list[int]] = {}
    for i, j, d in world:
        skeleton.setdefault(i, []).append(j)
        skeleton.setdefault(j, []).append(i)
        if d:
            directed.setdefault(i, []).append(j)

    # ---- model-implied reliance: intervention contrast per feature
    rng = np.random.default_rng(seed)
    anchor_idx = rng.choice(len(bg), size=min(n_anchor, len(bg)),
                            replace=False)

    def _contrast_shares(predict_fn) -> np.ndarray:
        lo_q, hi_q = quantiles
        rel = np.zeros(len(feats))
        for j in range(len(feats)):
            lo, hi = (np.percentile(bg[:, j], lo_q * 100),
                      np.percentile(bg[:, j], hi_q * 100))
            if hi - lo < 1e-12:
                continue
            A = bg[anchor_idx].copy()
            B = bg[anchor_idx].copy()
            A[:, j] = hi
            B[:, j] = lo
            rel[j] = float(np.mean(np.abs(predict_fn(A) - predict_fn(B))))
        return rel / (rel.sum() + 1e-12)

    shares = _contrast_shares(driver.predict)

    # ---- permutation null: refit on shuffled y, reliance becomes chance
    perm = None
    if n_perm and n_perm > 0:
        try:
            from sklearn.base import clone

            base = clone(driver.model)
            null = np.zeros((n_perm, len(feats)))
            for k in range(n_perm):
                mk = clone(base).fit(bg, rng.permutation(y))
                null[k] = _contrast_shares(mk.predict)
            perm = {
                "n_perm": n_perm,
                "null_share_median": np.median(null, axis=0),
                "p_value": (1.0 + (null >= shares).sum(axis=0)) / (n_perm + 1),
            }
        except Exception:  # noqa: BLE001 - permutation is optional evidence
            perm = None

    # ---- classify each relied upon feature against the world graph
    rows = []
    at_risk = 0.0
    order = np.argsort(-shares)
    for j in order:
        share = float(shares[j])
        if share < min_reliance and share < 1e-6:
            continue
        node = remap.get(j, j)
        if _directed_path(directed, node, outcome_idx):
            cls = "CAUSAL PATH"
        elif _directed_path(directed, outcome_idx, node):
            cls = "PROXY"
        elif _directed_path(skeleton, node, outcome_idx):
            cls = "UNRESOLVED PATH"
        else:
            cls = "INDEPENDENT"
        rows.append({
            "feature": feats[j],
            "reliance_share": round(share, 4),
            "classification": cls if share >= min_reliance else "BELOW THRESHOLD",
            **({"null_share_median": round(float(perm["null_share_median"][j]), 4),
                "perm_p": round(float(perm["p_value"][j]), 3)}
               if perm else {}),
        })
        if share >= min_reliance and cls != "CAUSAL PATH":
            at_risk += share

    risky = [r for r in rows if r["classification"] not in ("CAUSAL PATH", "BELOW THRESHOLD")]
    if risky:
        detail = ", ".join(
            f"{r['feature']} ({r['classification'].lower()}, "
            f"{r['reliance_share']:.0%}"
            + (f", perm p={r['perm_p']:.2f}" if "perm_p" in r else "")
            + ")" for r in risky)
        verdict = (
            f"SHORTCUT RISK: {at_risk:.0%} of interventional reliance sits on "
            f"features without a directed causal path to the outcome: {detail}"
        )
    else:
        verdict = (
            f"SUPPORTED: {1 - at_risk:.0%} of interventional reliance sits on "
            "features with a directed causal path to the outcome in the "
            f"{method} graph"
        )

    world_out = [
        {"source": names_kept[i], "target": names_kept[j],
         "directed": bool(d)}
        for i, j, d in sorted(world, key=lambda e: (e[0], e[1]))
    ]

    return {
        "n_features": len(feats),
        "discovery_method": method,
        "world_edges": world_out,
        "dropped_columns": dropped,
        "reliance": rows,
        "reliance_at_risk": round(at_risk, 4),
        "permutation": ({"n_perm": perm["n_perm"]} if perm else None),
        "verdict": verdict,
        "method": "world_graph_vs_interventional_reliance",
        "reference": (
            "Bellamy et al. (2022) structural characterization of shortcut "
            "features with causal DAGs; Sahidullah et al. (2025) shortcut "
            "discovery in black box classifiers; Geirhos et al. (2020) "
            "shortcut learning; PC (Spirtes and Glymour) or GES (Chickering) "
            "or DirectLiNGAM (Shimizu) for the world graph"
        ),
        "note": (
            "model reliance is the mean absolute p10 to p90 intervention "
            "contrast over anchor rows; the world graph is a CPDAG whose "
            "undirected edges are assumptions, user edges can pin them; "
            "PROXY features need deployment timing knowledge to separate "
            "label leakage from legitimate pre-outcome measurement"
        ),
    }
