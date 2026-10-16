"""因果图的人工审查与修正闭环。

流程：
  1. export_graph：自动发现 → 导出人类可编辑的 JSON（含内嵌说明）
  2. 领域专家编辑：确认/否决边、新增先验边、写批注
  3. apply_corrections：回灌 → 校验特征合法性 → **新增边自动过干预实验**
     （人提供先验，实验裁决真假——文献先验边的落地方刺）→ 输出 diff 报告

分工纪律：专家的否决/确认作为最高裁决被记录（含其 overridden 的自动证据）；
专家的新增边不被直接采信，必须通过 do 实验验证——先验与实验各司其职。
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np


def export_graph(session, path: str, sample_index: int | None = None) -> dict:
    from ..analysis.causal_graph import build_causal_graph

    g = build_causal_graph(session.driver,
                           session.driver.sample_at(sample_index) if sample_index is not None
                           else np.median(session.driver.background, axis=0))
    doc = {
        "_说明": (
            "审讯因果图：把边的 status 从 auto 改为 confirmed（确认）或 rejected（否决）；"
            "新增边请添加 {source, target, status: 'added', note: '你的依据'}——"
            "target 可为 'prediction'（特征→预测）或另一特征名（特征间影响）。"
            "新增边回灌时会被干预实验自动验证。编辑后运行 mic graph-apply。"
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
    """新增 f→prediction 边的干预验证：分位扫描实测预测漂移。"""
    drv = session.driver
    x = np.median(drv.background, axis=0)
    p_base = drv.predict_one(x)
    drifts = []
    for q in (0.1, 0.3, 0.5, 0.7, 0.9):
        drifts.append(abs(drv.predict_one(drv.intervene(x, {feature: drv.quantile_of(feature, q)})) - p_base))
    effect = float(np.mean(drifts))
    # 噪声底：用预测分布 std 的 5% 做显著性参照
    noise = float(np.std(drv.predict(drv.background[:200]))) * 0.05 + 1e-6
    return {"effect": round(effect, 5), "noise_floor": round(noise, 5),
            "verdict": "实验支持" if effect > noise else "实验不支持（效应低于噪声底）"}


def _verify_edge_feature_to_feature(session, source: str, target: str) -> dict:
    """新增 f→g 边的干预验证：do(f:=分位) 后 g 的归因漂移。"""
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
            "verdict": "实验支持" if effect > median_abs * 0.2 else "实验不支持（漂移微弱）"}


def apply_corrections(session, path: str) -> dict:
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
                results["invalid"].append({"edge": f"{src}->{tgt}", "reason": "特征不存在", "status": status})
                continue
            final_edges.append(e)
            if status == "confirmed":
                results["confirmed"].append(f"{src}->{tgt}")
            elif status == "rejected":
                results["rejected"].append(
                    {"edge": f"{src}->{tgt}", "note": e.get("note", ""),
                     "auto_evidence": f"自动效应 {e.get('effect')}（专家否决优先，已被记录）"}
                )
        elif status == "added":
            if src not in feats or (tgt != "prediction" and tgt not in feats):
                results["invalid"].append({"edge": f"{src}->{tgt}", "reason": "特征不存在", "status": "added"})
                continue
            verification = (
                _verify_edge_to_prediction(session, src) if tgt == "prediction"
                else _verify_edge_feature_to_feature(session, src, tgt)
            )
            entry = {"edge": f"{src}->{tgt}", "note": e.get("note", ""), "verification": verification}
            results["added"].append(entry)
            if verification["verdict"].startswith("实验支持"):
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
    n_ok = sum(1 for a in results["added"] if a["verification"]["verdict"].startswith("实验支持"))
    results["summary"] = (
        f"确认 {len(results['confirmed'])} 条 | 否决 {len(results['rejected'])} 条 | "
        f"新增 {n_add} 条（{n_ok} 条通过干预验证）| 非法 {len(results['invalid'])} 条"
    )
    return results
