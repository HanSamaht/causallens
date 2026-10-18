"""Causal graph prior injection for the investigation loop.

The investigator (an LLM or a human) arrives with domain knowledge
about how the features should relate. Left implicit, that knowledge
silently biases the reading of every tool output. This module makes
the prior a first-class object:

  1. set_causal_prior records the proposed edges on the tracker
  2. the prior nudges hypothesis log-odds at formation time, bounded
     by ln(1.5) so it can orient the investigation but never
     outweigh tool evidence, with every nudge logged
  3. test_causal_prior checks each prior edge against the intervention
     causal graph and returns supported, refuted, reversed, violated,
     or untested per edge

Prior edges are hypotheses like any other: they earn trust only by
surviving interventions.
"""
from __future__ import annotations

STRENGTHS = {"weak", "medium", "strong"}
SHARE_SUPPORT = 0.05   # direct effect share that counts as support
SHARE_REFUTE = 0.01    # below this the edge is considered refuted


def _validate_edges(edges, feature_names: list[str]) -> None:
    if not edges:
        raise ValueError("prior needs at least one edge")
    for e in edges:
        for key in ("source", "target"):
            v = e.get(key)
            if not isinstance(v, str) or not v:
                raise ValueError(f"prior edge needs a non-empty {key}")
        if e.get("target") != "prediction" and e["target"] not in feature_names:
            raise ValueError(f"unknown prior target {e['target']!r}")
        if e["source"] not in feature_names:
            raise ValueError(f"unknown prior source {e['source']!r}")
        s = e.get("strength", "medium")
        if s not in STRENGTHS:
            raise ValueError(f"strength must be one of {sorted(STRENGTHS)}, got {s!r}")


def set_causal_prior(session, edges: list[dict]) -> dict:
    """Record a causal prior and nudge already formed hypotheses.

    Each edge: {"source": feature, "target": feature or "prediction",
    "strength": "weak"|"medium"|"strong", "expected": true or false}.
    expected=false marks an edge the prior says should not exist.
    """
    _validate_edges(edges, session.driver.feature_names)
    session.investigation.set_prior(edges)
    injection = {"n_nudged": 0, "max_nudge": None}
    if session.investigation.hypotheses:
        injection = session.investigation.inject_prior()
    return {
        "n_edges": len(edges),
        "edges": edges,
        "hypothesis_injection": injection,
        "note": (
            "prior stored; call test_causal_prior to check each edge "
            "against intervention evidence"
        ),
    }


def test_causal_prior(session, sample_index: int | None = None) -> dict:
    """Check every prior edge against the intervention causal graph."""
    prior = session.investigation.causal_prior
    if not prior:
        raise ValueError("no prior stored; call set_causal_prior first")

    import numpy as np

    from ..analysis.causal_graph import build_causal_graph

    anchor = (
        session.driver.sample_at(sample_index)
        if sample_index is not None
        else np.median(session.driver.background, axis=0)
    )
    graph = build_causal_graph(session.driver, anchor)

    direct_share = {}
    total_direct = sum(e.effect for e in graph.edges if e.kind == "direct") or 1.0
    for e in graph.edges:
        if e.kind == "direct":
            direct_share[e.source] = e.effect / total_direct

    inter = [e for e in graph.edges if e.kind in ("interaction", "discovered")]

    rows = []
    for e in prior:
        src, expected = e["source"], e.get("expected", True)
        if e["target"] == "prediction":
            share = direct_share.get(src, 0.0)
            if expected:
                verdict = "supported" if share >= SHARE_SUPPORT else (
                    "refuted" if share < SHARE_REFUTE else "weak"
                )
                detail = f"direct effect share {share:.1%}"
            else:
                verdict = "violated" if share >= SHARE_SUPPORT else "clean"
                detail = f"model places {share:.1%} on a feature the prior banned"
        else:
            forward = [g for g in inter if g.source == src and g.target == e["target"]]
            backward = [g for g in inter if g.source == e["target"] and g.target == src]
            if forward:
                verdict, detail = "supported", "intervention edge found in the prior direction"
            elif backward:
                verdict, detail = "reversed", "intervention edge exists but points the other way"
            elif expected is False:
                verdict, detail = "clean", "no intervention edge, as the prior expects"
            else:
                verdict, detail = "untested", "no intervention edge detected in either direction"
            share = forward[0].effect if forward else (backward[0].effect if backward else 0.0)
        rows.append({
            "edge": f"{src} -> {e['target']}",
            "strength": e.get("strength", "medium"),
            "expected": expected,
            "verdict": verdict,
            "detail": detail,
            "matched_effect": round(float(share), 5),
        })

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    n_bad = counts.get("refuted", 0) + counts.get("violated", 0)
    if n_bad == 0 and counts.get("supported", 0) > 0:
        verdict = (
            f"PRIOR SURVIVES: {counts.get('supported', 0)} supported, "
            "no refuted or violated edges"
        )
    elif n_bad > 0:
        verdict = (
            f"PRIOR PARTIALLY WRONG: {n_bad} refuted or violated, "
            f"{counts.get('supported', 0)} supported"
        )
    else:
        verdict = "PRIOR UNTESTED: no intervention evidence matched any edge"

    return {
        "n_edges": len(rows),
        "rows": rows,
        "counts": counts,
        "verdict": verdict,
        "method": "prior_vs_intervention_graph",
        "note": (
            "untested means the intervention sweep found no edge in either "
            "direction, which is absence of evidence, not evidence of absence"
        ),
    }
