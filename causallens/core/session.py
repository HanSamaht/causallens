"""MicSession: interrogation API.

CLI(`mic report / audit / chat`), LLM (tool), MCP Server( LLM )
method——"model"from, evidenceconsistent.

 LLM also: offline_report / audit outputChinesereport.
"""
from __future__ import annotations

import numpy as np

from ..analysis import (
    aggregate_faithfulness,
    build_causal_graph,
    default_threshold,
    evaluate_faithfulness,
    explain_sample,
    explain_simulate_score,
    search_counterfactuals,
)
from ..analysis.drift import check_drift
from ..analysis.symbolic import symbolic_distill
from ..drivers.base import MicDriver
from .concepts import ConceptPack


class MicSession:
    """Micsession."""
    def __init__(self, driver: MicDriver, pack: ConceptPack | None = None):
        """Init."""
        self.driver = driver
        self.pack = pack or ConceptPack()
        self.evidence_log: list[dict] = []
        self.datasets: dict[str, "np.ndarray"] = {}
        from ..agents.investigator import HypothesisTracker

        self.investigation = HypothesisTracker()
        self.ledger = None # evidence ledger(optional, attach_ledger )
        self.case_id = None

    # ================= evidence ledger() =================

    def attach_ledger(self, ledger, model_version: str = "v1") -> dict:
        """/case archive: allevidence. """
        self.ledger = ledger
        self.case_id = ledger.open_case(self.driver.name, model_version)
        return self._log("attach_ledger", {"case_id": self.case_id,
                                           "model": self.driver.name, "model_version": model_version})

    def case_briefing(self) -> dict:
        """history: modelhistoryhypothesis(asinvestigationprior). """
        if self.ledger is None:
            return {"note": "(mic --ledger mic_ledger.db )"}
        return {
            "case_id": self.case_id,
            "chain": self.ledger.verify_chain(self.case_id),
            "priors": self.ledger.hypothesis_priors(self.driver.name),
        }

    # ================= Investigator (hypothesis + evidencethreshold) =================

    def form_hypotheses(self, statements: list[str]) -> dict:
        """Form hypotheses."""
        return self._log("form_hypotheses", self.investigation.form(statements))

    def get_hypotheses(self) -> dict:
        """Get hypotheses."""
        return self._log("get_hypotheses", self.investigation.snapshot())

    def update_hypotheses(self, updates: list[dict]) -> dict:
        """Update hypotheses."""
        return self._log("update_hypotheses", self.investigation.update(updates))

    def conclude_investigation(self) -> dict:
        """Conclude investigation."""
        v = self.investigation.verdict()
        if self.ledger is not None and self.case_id is not None:
            snap = self.investigation.snapshot()["hypotheses"]
            winner = v.get("winner") if v["verdict"] == "confirmed" else None
            for hid, h in snap.items():
                verdict = ("confirmed" if hid == winner else
                           "rejected" if v["verdict"] == "confirmed" else "inconclusive")
                self.ledger.record_hypothesis(
                    self.case_id, h["statement"], verdict,
                    h["posterior"] or 0.0,
                )
        return self._log("conclude_investigation", v)

    def investigate(self, n_samples: int = 10) -> dict:
        """investigator( LLM): fromfindinghypothesis, .

        upgrade: notis"audit "——everyfindingbecomeshypothesis,
        evidenceupdate, insufficientreport inconclusive.
        """
        audit = self.audit(None)
        codes = {f["code"]: f for f in audit["findings"]}

        # priorininvestigation(conclusion——historyinvestigation, )
        case_priors = None
        if self.ledger is not None:
            try:
                cur_ver = dict(self.ledger.conn.execute(
                    "SELECT model_version FROM cases WHERE id=?", (self.case_id,)
                ).fetchone())["model_version"]
                case_priors = [p for p in self.ledger.hypothesis_priors(self.driver.name)
                               if p["still_valid"] and p["model_version"] != cur_ver] or None
            except Exception:
                case_priors = None

        # ---- hypothesis(finding + concept packroletagged) ----
        statements = []
        if "spurious_dependence" in codes or "proxy_dependence" in codes:
            statements.append("model depends on concept-pack-tagged suspicious/proxy feature()")
        if "concentration" in codes:
            statements.append("modelconcentrated(feature)")
        if "faithfulness_profile" in codes or "faithfulness" in codes:
            statements.append("attribution explanations are not faithful(andmodel)")
        if "simulate_score" in codes:
            statements.append("SHAP-claimed importance inconsistent with intervention-measured sensitivity()")
        statements.append("model shows no obvious anomalies(onnot)")

        if len(statements) < 2:
            return {"verdict": "no_findings", "note": "hypothesis"}
        self.form_hypotheses(statements[:4])
        shares = audit["global_attribution_share"]

        # ---- hypothesis, update ----
        profile = self.run_faithfulness_profile(n_samples=n_samples)
        sim = self.simulate_score(0)

        updates = []
        if "spurious_dependence" in codes:
            f = codes["spurious_dependence"]
            updates.append(
                {
                    "hypothesis_id": self._match_hyp("suspicious"),
                    "tool_used": "audit",
                    "evidence_summary": f["message"],
                    "direction": "supports",
                    "weight": 3.0,
                }
            )
        if "concentration" in codes:
            top_share = list(shares.values())[0]
            updates.append(
                {
                    "hypothesis_id": self._match_hyp(""),
                    "tool_used": "audit.global_share",
                    "evidence_summary": f"Topfeatureattribution {top_share:.0%}",
                    "direction": "supports",
                    "weight": 2.0,
                }
            )
        # evidence: (support"not"hypothesis"")
        hyp_unfaith = self._match_hyp("not")
        hyp_clean = self._match_hyp("obvious")
        if profile["verdict"] == "":
            updates.append(
                {
                    "hypothesis_id": hyp_unfaith,
                    "tool_used": "run_faithfulness_profile",
                    "evidence_summary": f"passed {profile['pass_rate']:.0%} < 80%",
                    "direction": "supports",
                    "weight": 2.0,
                }
            )
            updates.append(
                {
                    "hypothesis_id": hyp_clean,
                    "tool_used": "run_faithfulness_profile",
                    "evidence_summary": f"passed {profile['pass_rate']:.0%} < 80%",
                    "direction": "against",
                    "weight": 2.0,
                }
            )
        else:
            updates.append(
                {
                    "hypothesis_id": hyp_clean,
                    "tool_used": "run_faithfulness_profile",
                    "evidence_summary": f"passed {profile['pass_rate']:.0%} ≥ 80%",
                    "direction": "supports",
                    "weight": 1.5,
                }
            )
        if sim["spearman"] is not None:
            d = "supports" if sim["spearman"] < 0.3 else "against"
            updates.append(
                {
                    "hypothesis_id": self._match_hyp("interventionnotconsistent"),
                    "tool_used": "simulate_score",
                    "evidence_summary": f"Spearman={sim['spearman']} verdict={sim['verdict']}",
                    "direction": d,
                    "weight": 2.0,
                }
            )
        # hypothesisupdate()
        valid = {h.hid for h in self.investigation.hypotheses.values()}
        updates = [u for u in updates if u["hypothesis_id"] in valid]
        if updates:
            self.update_hypotheses(updates)

        result = {
            "hypotheses": self.investigation.snapshot(),
            "verdict": self.conclude_investigation(),
            "audit_findings": [f["message"] for f in audit["findings"]],
        }
        if case_priors:
            result["case_priors"] = case_priors # historyhypothesisprior(, investigation)
        return result

    def _match_hyp(self, keyword: str) -> str:
        """Match hyp."""
        for h in self.investigation.hypotheses.values():
            if keyword in h.statement:
                return h.hid
        return next(iter(self.investigation.hypotheses), "H1")

    def _log(self, tool: str, payload: dict) -> dict:
        """Log tool call to evidence log."""
        """Log."""
        """Log tool call to evidence log."""
        self.evidence_log.append({"tool": tool, "payload": payload})
        if self.ledger is not None and self.case_id is not None:
            try:
                self.ledger.append_evidence(self.case_id, tool, payload)
            except Exception:
                pass # notinterrogation(evidencein)
        return payload

    # ================= interrogationtool(LLM / MCP alsoonlyallowthis) =================

    def set_target_class(self, k: int) -> dict:
        """interrogation: switch(attribution/counterfactualto k ), . """
        self.driver.target_class = int(k)
        if hasattr(self.driver, "_mic_explainer"):
            self.driver._mic_explainer = None
        return self._log("set_target_class", {"target_class": int(k), "n_classes": getattr(self.driver, "n_classes", 2)})

    def describe_model(self) -> dict:
        """Describe model."""
        d = self.driver.describe()
        d["domain"] = self.pack.domain
        d["concept_pack"] = self.pack.name or "(loading, featureconcept)"
        d["n_classes"] = getattr(self.driver, "n_classes", 2)
        d["target_class"] = getattr(self.driver, "target_class", 1)
        return self._log("describe_model", d)

    def list_samples(self, n: int = 5) -> dict:
        """predictionsample: most + most + sample. """
        p = self.driver.predict(self.driver.background)
        order_hi = np.argsort(-p)
        order_lo = np.argsort(p)
        idxs, seen = [], set()
        for cand in list(order_hi[: n // 2 + 1]) + list(order_lo[: n // 2 + 1]) + [int(np.argmin(abs(p - 0.5)))]:
            if cand not in seen:
                idxs.append(int(cand))
                seen.add(cand)
            if len(idxs) >= n:
                break
        return self._log(
            "list_samples",
            {"samples": [{"index": int(i), "prediction": round(float(p[i]), 4)} for i in idxs]},
        )

    def get_attribution(self, sample_index: int) -> dict:
        """Get attribution."""
        x = self.driver.sample_at(sample_index)
        p = self.driver.predict_one(x)
        attrs = explain_sample(self.driver, x, self.pack)
        total = sum(abs(a.shap_value) for a in attrs) or 1.0
        return self._log(
            "get_attribution",
            {
                "sample_index": int(sample_index),
                "prediction": round(p, 4),
                "semantics": self.driver.prediction_semantics,
                "attributions": [
                    {
                        "feature": a.feature,
                        "concept": a.concept,
                        "value": round(a.feature_value, 4),
                        "shap": round(a.shap_value, 4),
                        "share": round(abs(a.shap_value) / total, 4),
                        "role": a.role,
                    }
                    for a in attrs
                ],
            },
        )

    def intervene(self, sample_index: int, assignments: dict[str, float]) -> dict:
        """do(feature := value) prediction. causal graph, LLM . """
        x = self.driver.sample_at(sample_index)
        x2 = self.driver.intervene(x, assignments)
        return self._log(
            "intervene",
            {
                "sample_index": int(sample_index),
                "assignments": assignments,
                "prediction_before": round(self.driver.predict_one(x), 4),
                "prediction_after": round(self.driver.predict_one(x2), 4),
            },
        )

    def get_causal_graph(self, sample_index: int | None = None,
                         discovery: str | None = None) -> dict:
        """Get causal graph, optionally with causal-learn structure discovery."""
        anchor = (
            self.driver.sample_at(sample_index)
            if sample_index is not None
            else np.median(self.driver.background, axis=0)
        )
        g = build_causal_graph(self.driver, anchor, discovery=discovery)
        edges = [
            e.to_dict() | {"concept": self.pack.concept_of(e.source) if e.kind not in ("correlational", "discovered") else ""}
            for e in g.edges
        ]
        return self._log(
            "get_causal_graph",
            {
                "anchor_sample": sample_index,
                "direct_edges": [e for e in edges if e["kind"] == "direct"],
                "interaction_edges": [e for e in edges if e["kind"] == "interaction"],
                "correlational_edges": [e for e in edges if e["kind"] == "correlational"],
                "discovered_edges": [e for e in edges if e["kind"] == "discovered"],
                "method": g.method,
            },
        )

    def run_counterfactual(self, sample_index: int) -> dict:
        """Run counterfactual."""
        x = self.driver.sample_at(sample_index)
        paths = search_counterfactuals(self.driver, x, self.pack)
        threshold = default_threshold(self.driver)
        out_paths = []
        for cf in paths:
            out_paths.append(
                {
                    "changed_features": {
                        f: {
                            "concept": self.pack.concept_of(f.split("@")[0]),
                            "from": round(v[0], 4),
                            "to": round(v[1], 4),
                        }
                        for f, v in cf.changed_features.items()
                    },
                    "prediction_after": round(cf.cf_prediction, 4),
                    "distance": round(cf.distance, 4),
                    "method": cf.method,
                }
            )
        best = paths[0] if paths else None
        return self._log(
            "run_counterfactual",
            {
                "sample_index": int(sample_index),
                "threshold": threshold,
                "base_prediction": round(self.driver.predict_one(x), 4),
                "counterfactual_prediction": round(best.cf_prediction, 4) if best else None,
                "changed_features": out_paths[0]["changed_features"] if out_paths else {},
                "flipped": bool(best),
                "paths": out_paths,
            },
        )

    def ts_attribution(self, sample_index: int, granularity: str = "auto",
                       n_segments: int = 4, top_k: int = 10) -> dict:
        """attribution: channel(most) / segment() / full(most) / gradient() / auto(). """
        from ..analysis.granularity import timeseries_attribution

        assert self.driver.kind in ("timeseries", "torch"), "attributionrequire/torch "
        x = self.driver.sample_at(sample_index)
        result = timeseries_attribution(self.driver, x, granularity=granularity,
                                        n_segments=n_segments, top_k=top_k)
        result["sample_index"] = int(sample_index)
        return self._log("ts_attribution", result)

    def temporal_causal_graph(self, max_lag: int = 4, alpha: float = 0.05,
                              n_windows: int = 256, annotate_model: bool = True) -> dict:
        """Discover lagged channel causes with pairwise Granger tests.

        Each surviving edge channel j at lag k -> channel i at now is
        annotated with the model's attribution mass on that lag, so the
        report shows both what the data causes and what the model uses.
        """
        from ..analysis.temporal_causal import temporal_causal_graph

        assert self.driver.kind == "timeseries", "temporal discovery requires a timeseries driver"
        result = temporal_causal_graph(self.driver, max_lag=max_lag, alpha=alpha,
                                       n_windows=n_windows, annotate_model=annotate_model)
        return self._log("temporal_causal_graph", result)

    def fit_fastshap(self, n_samples: int = 150, epochs: int = 200) -> dict:
        """training FastSHAP —— full attributionfrom. """
        from ..analysis.fastshap import AmortizedExplainer
        from ..analysis.attribution import shap_matrix

        assert self.driver.kind in ("timeseries", "torch"), "FastSHAP require torch "
        n_feat = len(self.driver.feature_names)
        self._fastshap = AmortizedExplainer(n_feat)
        info = self._fastshap.fit(
            lambda X: shap_matrix(self.driver, X),
            self.driver.background, n_samples=n_samples, epochs=epochs,
        )
        return self._log("fit_fastshap", info)

    def fastshap_attribution(self, sample_index: int) -> dict:
        """attribution( fit_fastshap). """
        assert hasattr(self, "_fastshap") and self._fastshap.fitted, " fit_fastshap()"
        x = self.driver.sample_at(sample_index)
        sv = self._fastshap.attribute(x)
        T = getattr(self.driver, "seq_len", 1)
        F = getattr(self.driver, "n_features_raw", len(self.driver.feature_names))
        base = getattr(self.driver, "base_feature_names", self.driver.feature_names)
        from ..analysis.granularity import _aggregate_channel
        items = _aggregate_channel(sv, T, F, base)
        items.sort(key=lambda a: -abs(a["shap"]))
        return self._log("fastshap_attribution", {
            "engine": "FastSHAP(,)", "sample_index": int(sample_index),
            "attributions": [{**a, "role": "normal"} for a in items[:10]],
        })

    def run_faithfulness(self, sample_index: int) -> dict:
        """Run faithfulness."""
        x = self.driver.sample_at(sample_index)
        rep = evaluate_faithfulness(self.driver, x)
        return self._log("run_faithfulness", rep.to_dict())

    def run_faithfulness_profile(self, n_samples: int = 20) -> dict:
        """sample: single sample may be luck; pass rate shows overall trustworthiness. """
        return self._log("run_faithfulness_profile", aggregate_faithfulness(self.driver, n_samples=n_samples))

    def simulate_score(self, sample_index: int) -> dict:
        """explain-simulate-score: SHAP vs intervention experiment, Spearman . """
        x = self.driver.sample_at(sample_index)
        result = explain_simulate_score(self.driver, x)
        result["sample_index"] = int(sample_index)
        return self._log("simulate_score", result)

    # ================= tool: drift / backgroundupdate =================

    def register_data(self, name: str, data) -> dict:
        """(background/). not——LLM . """
        import numpy as np

        self.datasets[name] = np.asarray(data, dtype=float)
        return self._log("register_data", {"name": name, "shape": list(self.datasets[name].shape)})

    def list_datasets(self) -> dict:
        """List datasets."""
        return self._log(
            "list_datasets",
            {"datasets": {k: list(v.shape) for k, v in getattr(self, "datasets", {}).items()}},
        )

    def check_drift(self, new_data=None, data_ref: str | None = None) -> dict:
        """backgrounddistribution vs . driftby - SHAP .

        data_ref : register_data ; new_data as.
        """
        import numpy as np

        if data_ref is not None:
            new = self.datasets[data_ref]
        elif new_data is not None:
            new = np.asarray(new_data, dtype=float)
        else:
            raise ValueError("require data_ref or new_data ")
        return self._log("check_drift", check_drift(self.driver, new))

    def update_background(self, new_data) -> dict:
        """driftconfirm: replace/backgrounddistribution(). """
        import numpy as np

        new = np.asarray(new_data, dtype=float)
        self.driver.background = np.vstack([self.driver.background, new])[-2000:]
        return self._log(
            "update_background",
            {"n_background": len(self.driver.background), "note": "backgroundupdate, attribution/counterfactualdistribution"},
        )

    # ================= analysis: symbolic distillation / activationcausal graph / repair =================

    def symbolic_distill(self, n_samples: int = 400) -> dict:
        """symbolic distillation: modelbehavioraudit(PySR optional, Lasso ). """
        return self._log("symbolic_distill", symbolic_distill(self.driver, n_samples=n_samples))

    def has_torch_internals(self) -> bool:
        """Has torch internals."""
        return self.driver.kind in ("torch", "timeseries")

    def activation_graph(self, sample_index: int, reference_index: int) -> dict:
        """activationcausal graph(circuit tracing lite): layer+unit, locate. """
        from ..analysis.activation_graph import build_activation_graph

        assert self.has_torch_internals(), "activationcausal graphwhensupport torch "
        result = build_activation_graph(self.driver, sample_index, reference_index)
        result["sample_index"], result["reference_index"] = int(sample_index), int(reference_index)
        return self._log("activation_graph", result)

    def neuron_ranking(self, sample_index: int, layer: str | None = None, top_k: int = 10) -> dict:
        """ranking(Captum optional, fallback)——unitinthis. """
        from ..analysis.neuron_rank import neuron_ranking as _rank

        result = _rank(self.driver, sample_index, layer=layer, top_k=top_k)
        result["sample_index"] = int(sample_index)
        return self._log("neuron_ranking", result)

    def param_ablation(self, param_groups: list[str], metric: str = "pred_center",
                       n_samples: int = 50) -> dict:
        """reversible ablation: parameter(---recovery-). """
        from ..analysis.param_ablation import param_ablation as _pa

        result = _pa(self.driver, param_groups, metric=metric, n_samples=n_samples)
        return self._log("param_ablation", result)

    def discover_circuit(self, capability_samples: int = 20, tolerance: float = 0.02,
                         layer: str | None = None, mode: str = "greedy",
                         refinement_rounds: int = 3) -> dict:
        """ACDC-lite: mostunitcircuit. mode='greedy'() | 'exhaustive'(, ). """
        from ..analysis.param_ablation import discover_circuit as _dc

        result = _dc(self.driver, capability_samples=capability_samples,
                     tolerance=tolerance, layer=layer, mode=mode,
                     refinement_rounds=refinement_rounds)
        return self._log("discover_circuit", result)

    def visualize(self, sample_index: int = 0, output_path: str = "mic_report.html",
                  expand: int = 4, circuit_mode: str = "greedy") -> str:
        """Neuronpedia HTML interrogationreport(, ). """
        from ..analysis.visualize import generate_html

        path = generate_html(self, sample_index=sample_index, output_path=output_path)
        self._log("visualize", {"path": path})
        return path

    def sae_features(self, top_k: int = 8, expand: int = 4, epochs: int = 300) -> dict:
        """SAE-lite: + tagged + causal. expand=4/8/16(more=more). """
        from ..analysis.sae_lite import sae_features as _sf

        result = _sf(self.driver, top_k=top_k, expand=expand, epochs=epochs)
        return self._log("sae_features", result)

    def steering_repair(self, sample_index: int | None, feature: str, layer: str | None = None) -> dict:
        """repair: steering vector suspiciousfeature, interrogationtoolfollow-up. """
        from ..analysis.steering import steering_repair as _repair

        assert self.has_torch_internals(), "repairwhensupport torch "
        result = _repair(self, feature, layer=layer)
        result["sample_index"] = sample_index
        return self._log("steering_repair", result)

    # ================= counterfactual diversity =================

    def diverse_counterfactuals(self, sample_index: int, n_paths: int = 3) -> list:
        """Generate multiple distinct counterfactual paths (DiCE-style diversity)."""
        from ..analysis.counterfactual_diverse import diverse_counterfactuals
        x = self.driver.sample_at(sample_index)
        paths = diverse_counterfactuals(self.driver, x, self.pack, n_paths=n_paths)
        self._log("diverse_counterfactuals", {"sample_index": sample_index, "paths": paths})
        return paths

    def circuit_with_confidence(self, n_bootstrap: int = 20, tolerance: float = 0.02,
                                 layer: str | None = None) -> dict:
        """Circuit discovery with bootstrap confidence intervals on unit membership."""
        from ..analysis.circuit_bootstrap import circuit_with_confidence as _boot
        assert self.has_torch_internals(), "Circuit discovery requires torch driver"
        result = _boot(self.driver, n_bootstrap=n_bootstrap, tolerance=tolerance, layer=layer)
        return self._log("circuit_with_confidence", result)

    def mediation_decomposition(self, sample_index: int, reference_index: int,
                                layer: str) -> dict:
        """Decompose prediction change into natural direct and indirect effects.

        TE = NDE + NIE + interaction (Pearl 2001).
        NDE: effect from changing input, holding mediator at reference.
        NIE: effect from changing mediator, holding input at reference.
        """
        from ..analysis.mediation import mediation_decomposition
        assert self.has_torch_internals(), "Mediation decomposition requires torch driver"
        result = mediation_decomposition(self.driver, sample_index, reference_index, layer)
        return self._log("mediation_decomposition", result)

    def counterfactual_fairness(self, sample_index: int, protected_feature: str,
                                contrast_values: list[float],
                                n_samples: int = 20) -> dict:
        """Audit counterfactual fairness: would the same individual get a
        different outcome if only the protected attribute changed?

        Uses intervene() to change only the protected feature while holding
        all other features fixed. A prediction flip indicates the model
        violates counterfactual fairness (Kusner et al. 2017).
        """
        from ..analysis.fairness import counterfactual_fairness_check

        rng = np.random.default_rng(0)
        indices = rng.choice(
            len(self.driver.background),
            size=min(n_samples, len(self.driver.background)),
            replace=False,
        ).tolist()
        if sample_index not in indices:
            indices[0] = sample_index

        result = counterfactual_fairness_check(
            self.driver, indices, protected_feature, contrast_values
        )
        return self._log("counterfactual_fairness", result)

    def fairness_path_decomposition(self, protected_feature: str,
                                    contrast_low: float, contrast_high: float,
                                    mediators: list[str] | None = None,
                                    n_samples: int = 30) -> dict:
        """Split the protected attribute effect into direct and indirect paths.

        Direct discrimination means the protected attribute moves the
        decision on its own. Indirect means the effect rides through a
        mediator feature, which may be a legitimate decision factor that
        deserves review rather than removal.
        """
        from ..analysis.fairness import fairness_path_decomposition

        rng = np.random.default_rng(0)
        indices = rng.choice(
            len(self.driver.background),
            size=min(n_samples, len(self.driver.background)),
            replace=False,
        ).tolist()
        result = fairness_path_decomposition(
            self.driver, indices, protected_feature,
            contrast_low, contrast_high, mediators=mediators,
        )
        return self._log("fairness_path_decomposition", result)

    def check_contamination(self, new_data=None, data_ref: str | None = None,
                            y_bg=None, y_new=None) -> dict:
        """trainingbackground vs : + (training/testcontamination). """
        import numpy as np

        from ..analysis.leakage import duplicate_overlap, label_distribution_check

        new = self.datasets[data_ref] if data_ref is not None else np.asarray(new_data, dtype=float)
        result = {"overlap": duplicate_overlap(self.driver.background, new)}
        if y_bg is not None and y_new is not None:
            result["label_check"] = label_distribution_check(y_bg, y_new)
        return self._log("check_contamination", result)

    def leak_scan(self, y) -> dict:
        """featureleakagescan: background + - ' AUC'leakagefeature. """
        import numpy as np

        from ..analysis.leakage import single_feature_leak_scan

        y = np.asarray(y)
        if len(y) != len(self.driver.background):
            y = y[: len(self.driver.background)]
        return self._log("leak_scan", single_feature_leak_scan(y, self.driver.background,
                                                               self.driver.feature_names))

    # ================= evaluateinterrogation(BacktestDriver ) =================

    def has_backtest_internals(self) -> bool:
        """Has backtest internals."""
        return self.driver.kind == "backtest"

    def slippage_stress(self) -> dict:
        """test: PF . PF in bps = edge hypothesis. """
        assert self.has_backtest_internals(), "testrequire BacktestDriver"
        from ..backtest.stress import slippage_stress as _stress

        return self._log("slippage_stress", _stress(self.driver))

    def assumption_gap(self) -> dict:
        """: ladder price accounting vs market fill——backtesthypothesis. """
        assert self.has_backtest_internals(), "hypothesisrequire BacktestDriver"
        from ..backtest.stress import assumption_gap as _gap

        return self._log("assumption_gap", _gap(self.driver))

    def live_faithfulness(self, backtest: dict, live: dict) -> dict:
        """backtest vs ——evaluate(). """
        from ..backtest.stress import live_faithfulness as _lf

        return self._log("live_faithfulness", _lf(backtest, live))

    # ---- Torch tool(in) --------------------------------
    def has_activation_tools(self) -> bool:
        """Has activation tools."""
        return self.driver.kind in ("torch", "timeseries")

    def get_activations(self, sample_index: int) -> dict:
        """Get activations."""
        drv = self.driver
        assert drv.kind == "torch", "activationtoolsupport torch "
        x = drv.sample_at(sample_index)
        acts = drv.get_activations(x)
        return self._log(
            "get_activations",
            {
                "sample_index": int(sample_index),
                "layers": {k: {"shape": list(v.shape), "mean": round(float(v.mean()), 4)} for k, v in acts.items()},
            },
        )

    def patch_activation_test(self, sample_index: int, reference_index: int, layer: str) -> dict:
        """causalinanalysis: corrupt sample + layeractivation - outputrecovery. """
        drv = self.driver
        assert drv.kind == "torch", "activationtoolsupport torch "
        x = drv.sample_at(sample_index)
        x_ref = drv.sample_at(reference_index)
        p_clean = drv.predict_one(x)
        p_corrupt = drv.predict_one(x_ref)
        with drv.patch_activation(layer, x):
            p_patched = drv.predict_one(x_ref)
        return self._log(
            "patch_activation_test",
            {
                "layer": layer,
                "sample_index": int(sample_index),
                "reference_index": int(reference_index),
                "p_sample": round(p_clean, 4),
                "p_reference_baseline": round(p_corrupt, 4),
                "p_reference_patched": round(p_patched, 4),
                "mediation_effect": round(p_patched - p_corrupt, 4),
                "recovery_ratio": round((p_patched - p_corrupt) / (p_clean - p_corrupt), 4)
                if abs(p_clean - p_corrupt) > 1e-6 else None,
            },
        )

    # ================= interrogationreport( LLM, ) ========================

    def offline_report(self, sample_index: int) -> str:
        """Offline report."""
        attr = self.get_attribution(sample_index)
        cf = self.run_counterfactual(sample_index)
        faith = self.run_faithfulness(sample_index)
        graph = self.get_causal_graph(sample_index)

        d = self.driver
        sem = "probability" if d.prediction_semantics == "probability" else "score"
        lines = []
        lines.append("=" * 62)
        lines.append(f"MIC interrogationreport | model {d.name} ({d.kind}) | domain {self.pack.domain}")
        lines.append("=" * 62)
        p = attr["prediction"]
        verdict = "" if p > (0.5 if d.prediction_semantics == "probability" else (1.0 if d.kind == "backtest" else 0)) else ""
        label = "PF(>1as)" if d.kind == "backtest" else ("probability" if d.prediction_semantics == "probability" else "score")
        lines.append(f"sample #{sample_index}: prediction={verdict} ({label} {p:.3f})")
        lines.append("")
        lines.append("── attribution( |SHAP| descending, top 8)──")
        for a in attr["attributions"][:8]:
            flag = " ⚠️suspiciousfeature" if a["role"] == "spurious" else ""
            lines.append(
                f" {a['concept']:<12} value={a['value']:+.2f} contribution={a['shap']:+.3f} ( {a['share']:.0%}){flag}"
            )
        lines.append("")
        lines.append("── interventioncausal graph ──")
        direct = sorted(graph["direct_edges"], key=lambda e: -e["effect"])[:5]
        for e in direct:
            lines.append(f" [direct] {self.pack.concept_of(e['source'])} - prediction {e['detail']}")
        for e in graph["interaction_edges"][:4]:
            lines.append(
                f" [intervention] {self.pack.concept_of(e['source'])} - effect {self.pack.concept_of(e['target'])} attribution"
                f" (drift {e['effect']:.3f})"
            )
        for e in graph["correlational_edges"][:3]:
            lines.append(f" [correlation*] {e['source']} ~ {e['target']} (r={e['effect']:+.2f}) *observational only")
        lines.append("")
        lines.append("── counterfactual(most, most 3 )──")
        if cf["paths"]:
            for i, path in enumerate(cf["paths"], 1):
                changes = "; ".join(
                    f"{v['concept']} {v['from']:+.2f}-{v['to']:+.2f}" for v in path["changed_features"].values()
                )
                lines.append(
                    f" {i}: {changes} - prediction {path['prediction_after']:.3f}(, {path['distance']:.2f})"
                )
        else:
            lines.append(" inbackgrounddistributionsupportrangeto( ≤3 feature)")
        lines.append("")
        lines.append("── faithfulness verification(sample)──")
        f_ = faith
        status = "passed ✓" if f_["passed"] else "passed ✗"
        lines.append(
            f" [{status}] comprehensiveness delta={f_['comprehensiveness']:.3f} | random baseline delta={f_['random_baseline']:.3f}"
            f" | sufficiency delta={f_['sufficiency']:.3f} (top-k={f_['top_k']})"
        )
        for n in f_["notes"]:
            lines.append(f" - {n}")
        lines.append("")
        profile = self.run_faithfulness_profile(n_samples=12)
        sim = self.simulate_score(sample_index)
        lines.append("── (sample)──")
        lines.append(
            f" passed {profile['pass_rate']:.0%} | comprehensivenessvalue {profile['comprehensiveness']['mean']:.3f}"
            f"(p25-p75: {profile['comprehensiveness']['p25']:.3f}-{profile['comprehensiveness']['p75']:.3f})"
            f" | verdict: {profile['verdict']}"
        )
        lines.append(
            f" explain-simulate-score: Spearman={sim['spearman']} verdict={sim['verdict']}"
            f"(SHAP vs intervention)"
        )
        lines.append("")
        if self.driver.kind == "torch":
            lines.append("")
            lines.append("── activationinterrogation()──")
            try:
                acts = self.get_activations(sample_index)
                ref = (sample_index + 1) % len(self.driver.background)
                for layer in self.driver.list_layers()[:6]:
                    r = self.patch_activation_test(sample_index, ref, layer)
                    rec = r["recovery_ratio"]
                    lines.append(
                        f" layer[{layer}] activationvalue={acts['layers'][layer]['mean']:+.3f} "
                        f"inrecovery={rec if rec is None else round(rec, 2)}"
                    )
                lines.append(" (recovery=samplethelayeractivationsampleoutputsample; 1.0=thelayer)")
            except Exception as e:
                lines.append(f" activationinterrogationnot: {type(e).__name__}")

        findings = self.audit(sample_index)
        lines.append("── auditor ──")
        for f in findings["findings"]:
            lines.append(f" [{f['severity']}] {f['message']}")
        lines.append("=" * 62)
        return "\n".join(lines)

    # ================= auditor( + concept packroletagged) ==================

    def audit(self, sample_index: int | None = None) -> dict:
        """: interventionevidencescanmodel. """
        findings: list[dict] = []

        # 1) attributionconcentrated(backgroundsample)
        bg = self.driver.background[:200]
        from ..analysis.attribution import shap_matrix

        sm = shap_matrix(self.driver, bg)
        mean_abs = np.abs(sm).mean(axis=0)
        share = mean_abs / (mean_abs.sum() or 1.0)
        order = np.argsort(-share)
        feats = self.driver.feature_names

        if share[order[0]] > 0.5:
            findings.append(
                {
                    "severity": "warning",
                    "code": "concentration",
                    "message": (
                        f"featureconcentrated: '{self.pack.concept_of(feats[order[0]])}' "
                        f"attribution {share[order[0]]:.0%}, modelbehavior"
                    ),
                }
            )

        # 2) suspicious/proxyfeature(concept pack role tagged + attribution; taggedprior, 3%)
        for j, f in enumerate(feats):
            entry = self.pack.ground(f)
            if entry and entry.role in {"spurious", "proxy"} and share[j] > 0.03:
                findings.append(
                    {
                        "severity": "critical" if share[j] > 0.15 else "warning",
                        "code": f"{entry.role}_dependence",
                        "message": (
                            f"model{ 'suspicious' if entry.role == 'spurious' else 'proxy'}feature "
                            f"'{entry.concept}'({f}): attribution {share[j]:.0%}. "
                            f"this feature is tagged by concept pack as {entry.role}, verify its stability in deployment distribution"
                        ),
                    }
                )

        # 3) sample
        idx = sample_index if sample_index is not None else int(np.argmin(abs(share)))
        x = self.driver.sample_at(idx)
        rep = evaluate_faithfulness(self.driver, x)
        if not rep.passed:
            for n in rep.notes:
                findings.append({"severity": "warning", "code": "faithfulness", "message": f"sample #{idx}: {n}"})

        # 4) sample + explain-simulate-score
        prof = aggregate_faithfulness(self.driver, n_samples=12)
        if prof["verdict"] != "":
            findings.append(
                {
                    "severity": "warning",
                    "code": "faithfulness_profile",
                    "message": (
                        f"cross-sample faithfulness pass rate only {prof['pass_rate']:.0%}(threshold 80%)——"
                        f"themodelattribution explanations lack overall credibility; use with caution"
                    ),
                }
            )
        sim = explain_simulate_score(self.driver, x)
        if sim["verdict"] in {"/verdict", "partial"}:
            findings.append(
                {
                    "severity": "warning",
                    "code": "simulate_score",
                    "message": (
                        f"explain-simulate-score verdict'{sim['verdict']}'(Spearman={sim['spearman']}): "
                        f"SHAP featureandintervention experimentnotconsistent, "
                    ),
                }
            )

        if not findings:
            findings.append(
                {
                    "severity": "passed",
                    "code": "none",
                    "message": "findingobviousproblem: attribution, suspiciousfeature, checkpassed",
                }
            )

        return self._log(
            "audit",
            {
                "global_attribution_share": {
                    self.pack.concept_of(feats[j]): round(float(share[j]), 4) for j in order
                },
                "findings": findings,
            },
        )
