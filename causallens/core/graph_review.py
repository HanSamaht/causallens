"""causal graphand.

:
  1. export_graph:finding -> JSON()
  2. domain:confirm/,prior,
  3. apply_corrections: -> feature -> **intervention experiment**
     (prior,——prior)-> output diff report

:/confirmashighestbyrecord(its overridden evidence);
notbydirect,mustpassed do validate——priorandits.
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np


def export_graph(session, path: str, sample_index: int | None = None) -> dict:
    """Export graph."""
    from ..analysis.causal_graph import build_causal_graph

    g = build_causal_graph(session.driver,
                           session.driver.sample_at(sample_index) if sample_index is not None
                           else np.median(session.driver.background, axis=0))
    doc = {
        "_": (
            "interrogationcausal graph: status from auto as confirmed(confirm)or rejected();"
            "add {source, target, status: 'added', note: ''}——"
            "target as 'prediction'(feature->prediction)orfeature(featureeffect)."
            "byintervention experimentvalidate.run mic graph-apply."
        ),
        "model": session.driver.name,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "kind": e.kind,
                "effect": round(e.effect, 4),
                "status": "auto",
                "note": "",
            }
            for e in g.edges
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return {"path": path, "n_edges": len(doc["edges"]), "model": session.driver.name}


def _verify_edge_to_prediction(session, feature: str) -> dict:
    """ f->prediction interventionvalidate:percentilescanpredictiondrift."""
    drv = session.driver
    x = np.median(drv.background, axis=0)
    p_base = drv.predict_one(x)
    drifts = []
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        drifts.append(abs(drv.predict_one(drv.intervene(x, {feature: drv.quantile_of(feature, q)})) - p_base))
    effect = float(np.mean(drifts))
    # :predictiondistribution std 5% significant
    noise = float(np.std(drv.predict(drv.background[:200]))) * 0.05 + 1e-6
    return {"effect": round(effect, 5), "noise_floor": round(noise, 5),
            "verdict": "support" if effect > noise else "notsupport(effect)"}


def _verify_edge_feature_to_feature(session, source: str, target: str) -> dict:
    """ f->g interventionvalidate:do(f:=percentile) g attributiondrift."""
    from ..analysis.attribution import shap_matrix

    drv = session.driver
    x = np.median(drv.background, axis=0)
    base = shap_matrix(drv, x.reshape(1, -1))[0]
    j_src = drv.feature_names.index(source)
    j_tgt = drv.feature_names.index(target)
    drifts = []
    for q in (0.1, 0.5, 0.9):
        s2 = shap_matrix(drv, drv.intervene(x, {source: drv.quantile_of(source, q)}).reshape(1, -1))[0]
        drifts.append(abs(s2[j_tgt] - base[j_tgt]))
    effect = float(np.mean(drifts))
    median_abs = float(np.median(np.abs(base))) + 1e-6
    return {"effect": round(effect, 5), "reference_median_attribution": round(median_abs, 5),
            "verdict": "support" if effect > median_abs * 0.2 else "notsupport(drift)"}


def apply_corrections(session, path: str) -> dict:
    """Apply corrections."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    feats = set(session.driver.feature_names)
    results = {"confirmed": [], "rejected": [], "added": [], "invalid": []}
    final_edges = []

    for e in doc.get("edges", []):
        status = e.get("status", "auto")
        src, tgt = e.get("source", ""), e.get("target", "")

        if status in ("auto", "confirmed", "rejected"):
            if src not in feats or (tgt != "prediction" and tgt not in feats):
                results["invalid"].append({"edge": f"{src}->{tgt}", "reason": "featurenotin", "status": status})
                continue
            final_edges.append(e)
            if status == "confirmed":
                results["confirmed"].append(f"{src}->{tgt}")
            elif status == "rejected":
                results["rejected"].append(
                    {"edge": f"{src}->{tgt}", "note": e.get("note", ""),
                     "auto_evidence": f"effect {e.get('effect')}(,byrecord)"}
                )
        elif status == "added":
            if src not in feats or (tgt != "prediction" and tgt not in feats):
                results["invalid"].append({"edge": f"{src}->{tgt}", "reason": "featurenotin", "status": "added"})
                continue
            verification = (
                _verify_edge_to_prediction(session, src) if tgt == "prediction"
                else _verify_edge_feature_to_feature(session, src, tgt)
            )
            entry = {"edge": f"{src}->{tgt}", "note": e.get("note", ""), "verification": verification}
            results["added"].append(entry)
            if verification["verdict"].startswith("support"):
                e = dict(e, kind="user_prior_verified", effect=verification["effect"])
            else:
                e = dict(e, kind="user_prior_rejected", effect=verification["effect"])
            final_edges.append(e)

    session.corrected_graph = {"model": doc.get("model"), "applied_at": datetime.now().isoformat(timespec="seconds"),
                               "edges": final_edges}
    if getattr(session, "ledger", None) is not None and session.case_id is not None:
        for c in results["confirmed"]:
            session.ledger.record_correction(session.case_id, c, "confirmed", "", None)
        for r in results["rejected"]:
            session.ledger.record_correction(session.case_id, r["edge"], "rejected", r.get("note", ""), None)
        for a in results["added"]:
            session.ledger.record_correction(session.case_id, a["edge"], "added", a.get("note", ""),
                                             a.get("verification"))
    n_add = len(results["added"])
    n_ok = sum(1 for a in results["added"] if a["verification"]["verdict"].startswith("support"))
    results["summary"] = (
        f"confirm {len(results['confirmed'])} | {len(results['rejected'])} | "
        f" {n_add} ({n_ok} passedinterventionvalidate)| {len(results['invalid'])} "
    )
    return results
