"""Real world effect identification through DoWhy.

The framework's do() experiments interrogate the model: they say what
the model does when an input changes. They cannot say what the true
causal effect in the world is, because confounding lives in the data
generating process, not in the model. This module answers the real
effect question with the classical machinery: backdoor and front-door
criteria, via DoWhy.

Graph provenance, in priority order:
  1. caller supplied edges, for when the user knows the domain
  2. the Intervense intervention causal graph plus priors, where
     interaction and discovered edges become directed arrows and
     correlational edges become latent common causes (U nodes), which
     is the honest translation: correlation is unexplained, so it is
     modeled as an unobserved confounder

The identification verdict names the estimand (backdoor set,
front-door, instrumental) or reports not identified. An estimate from
an unidentified effect is refused, not guessed.
"""
from __future__ import annotations

import numpy as np

try:  # optional dependency, soft fallback
    from dowhy import CausalModel

    HAS_DOWHY = True
except Exception:  # pragma: no cover - exercised only without the library
    HAS_DOWHY = False


def _edges_from_framework(driver, max_samples: int) -> list[tuple[str, str, str]]:
    """(source, target, kind) triples from the intervention causal graph.

    The interaction drift sweep is symmetric: when f moves g's
    attribution and g moves f's, both arrows pass the threshold, but
    neither direction is established. Such reciprocal pairs collapse to
    one correlational edge, which downstream becomes a latent common
    cause. Discovered edges keep their orientation: a discovery backend
    that orients f -> g is evidence of direction.
    """
    from .causal_graph import build_causal_graph

    anchor = np.median(driver.background[:max_samples], axis=0)
    g = build_causal_graph(driver, anchor)

    directed_pairs: set[tuple[str, str]] = set()
    for e in g.edges:
        if e.kind in ("interaction", "discovered", "direct"):
            directed_pairs.add((e.source, e.target))
    reciprocal = {p for p in directed_pairs if (p[1], p[0]) in directed_pairs}

    triples: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for e in g.edges:
        if e.kind == "direct":
            key = (e.source, e.target, "directed")
        elif e.kind == "discovered":
            # discovered orientation overrides the reciprocal collapse
            key = (e.source, e.target, "directed")
        elif e.kind == "interaction":
            pair = (e.source, e.target)
            if pair in reciprocal:
                key = (min(pair), max(pair), "correlational")
            else:
                key = (e.source, e.target, "directed")
        elif e.kind == "correlational":
            key = (min(e.source, e.target), max(e.source, e.target), "correlational")
        else:
            continue
        if key not in seen:
            seen.add(key)
            triples.append(key)
    return triples


def _build_networkx(triples: list[tuple[str, str, str]]):
    import networkx as nx

    g = nx.DiGraph()
    counter = 0
    for src, dst, kind in triples:
        if src not in g:
            g.add_node(src)
        if dst not in g:
            g.add_node(dst)
        if kind == "directed":
            g.add_edge(src, dst)
        else:
            # correlation is modeled as a latent common cause
            u = f"U_conf_{counter}"
            counter += 1
            g.add_node(u, observed="no")
            g.add_edge(u, src)
            g.add_edge(u, dst)
    return g


def identify_real_effect(
    driver,
    treatment: str,
    outcome: str = "prediction",
    edges: list[tuple[str, str]] | None = None,
    n_samples: int = 500,
    estimate: bool = True,
) -> dict:
    """Identify and optionally estimate the real causal effect.

    Args:
        driver: any Intervense driver, the background is the data.
        treatment: feature name to intervene on in the world.
        outcome: "prediction" for the model output, or a feature name.
        edges: optional explicit [(source, target)] arrows. Correlation
            between the endpoints is assumed absent when supplied
            explicitly; supply U nodes yourself if you suspect latents.
        n_samples: rows of background used.
        estimate: compute the estimate when identified; identification
            alone is faster.

    Returns:
        Dict with identified flag, estimand description, backdoor set,
        estimate and refutation when available.
    """
    if not HAS_DOWHY:
        raise ImportError("DoWhy is not installed. Install it with: pip install dowhy")
    feats = list(driver.feature_names)
    if treatment not in feats:
        raise ValueError(f"unknown treatment {treatment!r}")
    if outcome != "prediction" and outcome not in feats:
        raise ValueError(f"unknown outcome {outcome!r}")

    if edges is not None:
        triples = [(s, t, "directed") for s, t in edges]
        provenance = "user supplied graph"
    else:
        triples = _edges_from_framework(driver, n_samples)
        provenance = "Intervense intervention graph, correlational edges as latent confounders"

    n = min(n_samples, len(driver.background))
    X = driver.background[:n]
    data = {f: X[:, j] for j, f in enumerate(feats)}
    if outcome == "prediction":
        data["prediction"] = driver.predict(X)
    import pandas as pd

    df = pd.DataFrame(data)

    graph = _build_networkx(triples)
    if treatment not in graph:
        graph.add_node(treatment)
    if outcome not in graph:
        graph.add_node(outcome)

    # strict latent confounding check, stricter than DoWhy's lenient
    # default: a latent U pointing into both treatment and outcome is an
    # unblockable backdoor path and the effect is refused here
    latent_confounded = any(
        treatment in graph.successors(u) and outcome in graph.successors(u)
        for u in graph.nodes
        if u.startswith("U_conf_")
    )

    model = CausalModel(
        data=df, treatment=treatment, outcome=outcome, graph=graph
    )
    def _not_identified(reason: str) -> dict:
        return {
            "treatment": treatment,
            "outcome": outcome,
            "graph_provenance": provenance,
            "n_samples": n,
            "identified": False,
            "estimand": reason,
            "backdoor_set": [],
            "estimate": None,
            "method": "dowhy_identification",
            "reference": "Sharma and Kiciman, DoWhy (2020)",
            "verdict": (
                "NOT IDENTIFIED: " + reason + "; no estimate is reported"
            ),
        }

    if latent_confounded:
        return _not_identified(
            "a latent common cause connects the treatment and the outcome "
            "in the causal graph and no observed variable can block it"
        )
    try:
        estimand = model.identify_effect(proceed_when_unidentifiable=False)
    except Exception as e:
        return _not_identified(f"identification failed: {type(e).__name__}: {e}")
    estimand_str = str(estimand)

    backdoor_set: list[str] = []
    try:
        bd = estimand.get_backdoor_variables()
        backdoor_set = list(bd) if bd else []
    except Exception:
        pass

    identified = len(estimand_str.strip()) > 0 and "no valid" not in estimand_str.lower()

    result = {
        "treatment": treatment,
        "outcome": outcome,
        "graph_provenance": provenance,
        "n_samples": n,
        "identified": identified,
        "estimand": estimand_str[:2000],
        "backdoor_set": backdoor_set,
        "estimate": None,
        "method": "dowhy_identification",
        "reference": "Sharma and Kiciman, DoWhy (2020)",
        "note": (
            "identification is only as good as the graph; the framework graph"
            if edges is None else
            "identification is only as good as the supplied graph"
        ),
    }

    control_value = float(np.percentile(df[treatment], 25))
    treatment_value = float(np.percentile(df[treatment], 75))
    result["treatment_contrast"] = {
        "from_p25": round(control_value, 4),
        "to_p75": round(treatment_value, 4),
    }
    if identified and estimate:
        try:
            method = "backdoor.linear_regression"
            est = model.estimate_effect(
                estimand, method_name=method,
                control_value=control_value,
                treatment_value=treatment_value,
                confidence_intervals=False,
            )
            result["estimate"] = round(float(est.value), 6)
            result["estimator"] = method
        except Exception as e:
            result["estimate_error"] = f"{type(e).__name__}: {e}"

    if identified and result.get("estimate") is not None:
        verdict = (
            f"IDENTIFIED via {('backdoor set ' + str(backdoor_set)) if backdoor_set else 'the graph'}"
            f", effect estimate {result['estimate']:+.4f}"
        )
    elif identified:
        verdict = (
            "IDENTIFIED in principle, estimation failed or skipped; "
            "inspect the estimand field"
        )
    else:
        verdict = (
            "NOT IDENTIFIED: the graph does not license this effect, "
            "unobserved confounding cannot be ruled out; no estimate is reported"
        )
    result["verdict"] = verdict
    return result
