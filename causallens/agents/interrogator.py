"""interrogation:LLM passedtoolmodel//recommend.

 LLM configuration,ask() as"report"mode,framework.
"""
from __future__ import annotations

import json

from ..core.session import MicSession
from .llm import LLMClient
from .prompts import AUDITOR, FIXER, TRANSLATOR

ROLE_PROMPTS = {"translator": TRANSLATOR, "auditor": AUDITOR, "fixer": FIXER}

MAX_TOOL_ROUNDS = 8


class Interrogator:
    """Interrogator."""
    def __init__(self, session: MicSession, llm: LLMClient | None = None, role: str = "translator"):
        """Init."""
        self.session = session
        self.llm = llm
        self.role = role if role in ROLE_PROMPTS else "translator"

    # ---- LLM tool(OpenAI function calling ) -------------
    def tool_schemas(self) -> list[dict]:
        """Tool schemas."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "describe_model",
                    "description": "byinterrogationmodel(type,feature,prediction)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_attribution",
                    "description": "samplepredictionandfeature SHAP attribution(conceptgroundingandroletagged)",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "intervene",
                    "description": "intervention experiment:featureset tovalue,prediction(do )",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sample_index": {"type": "integer"},
                            "assignments": {
                                "type": "object",
                                "description": "feature -> value, {\"momentum\": 1.5}",
                            },
                        },
                        "required": ["sample_index", "assignments"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_causal_graph",
                    "description": "causal graph:direct (SHAPeffect) + interaction (intervention experiment) + correlational (correlation)",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_counterfactual",
                    "description": "predictionmostfeature(counterfactual)",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_faithfulness",
                    "description": "faithfulness verification:comprehensiveness/sufficiency/random baseline",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "audit",
                    "description": (
                        "run(range:suspicious/proxyfeature,concentrated,sampleandsample,"
                        "explain-simulate-score). Not a distribution or OOD test, run "
                        "check_drift for that"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_faithfulness_profile",
                    "description": "sample(12-20 sample):passed,comprehensivenesspercentile,verdict",
                    "parameters": {
                        "type": "object",
                        "properties": {"n_samples": {"type": "integer", "description": "default 20"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "simulate_score",
                    "description": "explain->simulate->score:SHAP vs interventionSpearmanconsistent",
                    "parameters": {
                        "type": "object",
                        "properties": {"sample_index": {"type": "integer"}},
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_samples",
                    "description": "sample(most/most/),tointerrogationsample",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer", "description": "returncount,default 5"}},
                    },
                },
            },
        ]
        if self.session.has_activation_tools():
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": "get_activations",
                        "description": "sampleineverylayeractivation",
                        "parameters": {
                            "type": "object",
                            "properties": {"sample_index": {"type": "integer"}},
                            "required": ["sample_index"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "patch_activation_test",
                        "description": "activation patching(causalinanalysis):samplelayeractivationsample,outputrecovery",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sample_index": {"type": "integer", "description": "sample"},
                                "reference_index": {"type": "integer", "description": "bysample"},
                                "layer": {"type": "string"},
                            },
                            "required": ["sample_index", "reference_index", "layer"],
                        },
                    },
                },
            ]
        tools += [
            {
                "type": "function",
                "function": {
                    "name": "form_hypotheses",
                    "description": "form 2 to 4 competing hypotheses to open the investigation",
                    "parameters": {
                        "type": "object",
                        "properties": {"statements": {"type": "array", "items": {"type": "string"}}},
                        "required": ["statements"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_hypotheses",
                    "description": "hypothesiswhenandposteriorprobability(,not)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_hypotheses",
                    "description": "updatehypothesis:evidence_summarymust,direction=supports/against/neutral",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "updates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "hypothesis_id": {"type": "string"},
                                        "tool_used": {"type": "string"},
                                        "evidence_summary": {"type": "string"},
                                        "direction": {"type": "string", "enum": ["supports", "against", "neutral"]},
                                        "weight": {"type": "number", "description": "evidence0.2-5,default1"},
                                    },
                                    "required": ["hypothesis_id", "tool_used", "evidence_summary", "direction"],
                                },
                            }
                        },
                        "required": ["updates"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "conclude_investigation",
                    "description": "investigationconclusion:if odds ratio met, announce winner; otherwise honestly report inconclusive",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "conclude_investigation",
                    "description": "investigationconclusion:if odds ratio met, announce winner; otherwise honestly report inconclusive",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "plan_next_intervention",
                    "description": (
                        "rank candidate interventions by expected information "
                        "gain over the standing hypotheses; call this when "
                        "unsure which experiment to run next"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "specification_audit",
                    "description": (
                        "refit the model family on every control subset and "
                        "report whether the treatment's intervention effect "
                        "keeps its sign, which specifications flip it, and "
                        "the minimal control sets responsible; call this "
                        "before quoting any feature effect from a "
                        "multivariate model, collinearity can flip signs "
                        "that importance rankings hide; the training "
                        "outcome must be registered as dataset 'y' or "
                        "named via y_ref"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "treatment": {
                                "type": "string",
                                "description": "feature whose effect is audited",
                            },
                            "y_ref": {
                                "type": "string",
                                "description": (
                                    "registered dataset name holding the "
                                    "training outcome, default y"
                                ),
                            },
                        },
                        "required": ["treatment"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "set_causal_prior",
                    "description": (
                        "record your domain prior as explicit causal edges "
                        "(source feature, target feature or prediction, strength, "
                        "expected true or false); the prior nudges hypothesis "
                        "log-odds by at most ln(1.5) and every nudge is logged"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "edges": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "target": {"type": "string"},
                                        "strength": {"type": "string", "enum": ["weak", "medium", "strong"]},
                                        "expected": {"type": "boolean"},
                                    },
                                    "required": ["source", "target"],
                                },
                            }
                        },
                        "required": ["edges"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "test_causal_prior",
                    "description": (
                        "check every stored prior edge against the intervention "
                        "causal graph: supported, refuted, reversed, violated, "
                        "or untested"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "leak_scan",
                    "description": "single feature leakage scan, needs datasets['y']; flags features whose solo AUC rivals the model",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "symbolic_distill",
                    "description": "symbolic distillation:modelbehavioraudit('modelin'most)",
                    "parameters": {
                        "type": "object",
                        "properties": {"n_samples": {"type": "integer", "description": "default 400"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_drift",
                    "description": "distributiondrift(KS+PSI+predictiondrift). data_ref ( list_datasets )",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "data_ref": {"type": "string", "description": ", 'val_2026_10'"},
                            "new_data": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "number"}},
                                "description": "(20 data_ref)",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "register_data",
                    "description": "()tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "data": {"type": "array"}},
                        "required": ["name", "data"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_datasets",
                    "description": "(drift)",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        if self.session.has_backtest_internals():
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": "slippage_stress",
                        "description": "test:PF;PFinbpsto1='edgehypothesis'feature",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "assumption_gap",
                        "description": ":ladder price accounting vs market fill,backtesthypothesisPF",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "live_faithfulness",
                        "description": "backtestvs(:pf/win_rate/avg_pnl_pct)",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "backtest": {"type": "object"},
                                "live": {"type": "object"},
                            },
                            "required": ["backtest", "live"],
                        },
                    },
                },
            ]
        if self.session.has_torch_internals():
            tools += [
                {
                    "type": "function",
                    "function": {
                        "name": "param_ablation",
                    "description": "reversible ablation:parameter( gru.weight_hh is)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param_groups": {"type": "array", "items": {"type": "string"}},
                            "metric": {"type": "string", "enum": ["pred_center", "early_window_share"]},
                        },
                        "required": ["param_groups"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "discover_circuit",
                    "description": "ACDC-lite:whenbehaviormostunitcircuit()",
                    "parameters": {
                        "type": "object",
                        "properties": {"tolerance": {"type": "number"}},
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sae_features",
                    "description": "SAE-lite:trainingtaggedfeaturecausal('thisweightin')",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "neuron_ranking",
                    "description": "ranking(unitinthis;ranking=correlation, activation_graph causalconfirm)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sample_index": {"type": "integer"},
                            "layer": {"type": "string"},
                        },
                        "required": ["sample_index"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "activation_graph",
                        "description": "activationcausal graph(circuit tracing lite):layer+unitintervention,locate",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sample_index": {"type": "integer"},
                                "reference_index": {"type": "integer"},
                            },
                            "required": ["sample_index", "reference_index"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "steering_repair",
                        "description": "repair: steering vector suspiciousfeature(intervention),follow-upvalidate",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "sample_index": {"type": "integer"},
                                "feature": {"type": "string", "description": "suspiciousfeature"},
                                "layer": {"type": "string", "description": "layer,optional"},
                            },
                            "required": ["feature"],
                        },
                    },
                },
            ]
        return tools

    def _execute_tool(self, name: str, args: dict) -> dict:
        """Execute tool."""
        s = self.session
        dispatch = {
            "describe_model": lambda: s.describe_model(),
            "get_attribution": lambda: s.get_attribution(int(args["sample_index"])),
            "intervene": lambda: s.intervene(int(args["sample_index"]), args["assignments"]),
            "get_causal_graph": lambda: s.get_causal_graph(args.get("sample_index")),
            "run_counterfactual": lambda: s.run_counterfactual(int(args["sample_index"])),
            "run_faithfulness": lambda: s.run_faithfulness(int(args["sample_index"])),
            "audit": lambda: s.audit(args.get("sample_index")),
            "run_faithfulness_profile": lambda: s.run_faithfulness_profile(int(args.get("n_samples", 20))),
            "simulate_score": lambda: s.simulate_score(int(args["sample_index"])),
            "list_samples": lambda: s.list_samples(int(args.get("n", 5))),
            "get_activations": lambda: s.get_activations(int(args["sample_index"])),
            "patch_activation_test": lambda: s.patch_activation_test(
                int(args["sample_index"]), int(args["reference_index"]), args["layer"]
            ),
            "check_contamination": lambda: s.check_contamination(None, args.get("data_ref")),
            "leak_scan": lambda: s.leak_scan(s.datasets.get("y", [1] * len(s.driver.background))),
            "symbolic_distill": lambda: s.symbolic_distill(int(args.get("n_samples", 400))),
            "check_drift": lambda: s.check_drift(args.get("new_data"), args.get("data_ref")),
            "register_data": lambda: s.register_data(args["name"], args["data"]),
            "list_datasets": lambda: s.list_datasets(),
            "form_hypotheses": lambda: s.form_hypotheses(args["statements"]),
            "get_hypotheses": lambda: s.get_hypotheses(),
            "update_hypotheses": lambda: s.update_hypotheses(args["updates"]),
            "conclude_investigation": lambda: s.conclude_investigation(),
            "plan_next_intervention": lambda: s.plan_next_intervention(),
            "specification_audit": lambda: s.specification_audit(
                args["treatment"], s.datasets.get(args.get("y_ref", "y"))
            ),
            "set_causal_prior": lambda: s.set_causal_prior(args["edges"]),
            "test_causal_prior": lambda: s.test_causal_prior(),
            "slippage_stress": lambda: s.slippage_stress(),
            "assumption_gap": lambda: s.assumption_gap(),
            "live_faithfulness": lambda: s.live_faithfulness(args["backtest"], args["live"]),
            "param_ablation": lambda: s.param_ablation(args["param_groups"], args.get("metric", "pred_center")),
            "discover_circuit": lambda: s.discover_circuit(tolerance=float(args.get("tolerance", 0.02))),
            "sae_features": lambda: s.sae_features(),
            "neuron_ranking": lambda: s.neuron_ranking(int(args["sample_index"]), args.get("layer")),
            "activation_graph": lambda: s.activation_graph(
                int(args["sample_index"]), int(args["reference_index"])
            ),
            "steering_repair": lambda: s.steering_repair(
                args.get("sample_index"), args["feature"], args.get("layer")
            ),
        }
        if name not in dispatch:
            return {"error": f"tool {name}"}
        try:
            return dispatch[name]()
        except Exception as e: # tool LLM to,notin
            return {"error": f"{type(e).__name__}: {e}"}

    # ---- -------------------------------------------------------
    def ask(self, question: str) -> str:
        """Run tool-calling loop to answer a question about the model."""
        if self.llm is None:
            return self._offline_answer(question)

        messages = [
            {"role": "system", "content": ROLE_PROMPTS[self.role]},
            {
                "role": "user",
                "content": f"byinterrogationmodel:{json.dumps(self.session.describe_model(), ensure_ascii=False)}\n\nproblem:{question}",
            },
        ]
        tools = self.tool_schemas()
        start_log = len(self.session.evidence_log)

        for _ in range(MAX_TOOL_ROUNDS):
            resp = self.llm.chat(messages, tools=tools)
            msg = resp["choices"][0]["message"]
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # execute:investigatorhypothesisbelowconclusion ->
                if self.role == "auditor" and self.session.investigation.hypotheses:
                    concluded = any(
                        e["tool"] == "conclude_investigation"
                        for e in self.session.evidence_log[start_log:]
                    )
                    if not concluded:
                        messages.append({"role": "assistant", "content": msg.get("content") or ""})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "investigation:hypothesis. "
                                    "update_hypotheses;evidencemust conclude_investigation "
                                    "(notalsoallowreport inconclusive)."
                                ),
                            }
                        )
                        continue
                return msg.get("content") or "()"
            # tool_calls(consistent),executeversion
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
            from .llm import LLMClient

            for tc in LLMClient.normalize_tool_calls(msg):
                result = self._execute_tool(tc["name"], tc["arguments"])
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result, ensure_ascii=False, default=str)[:6000],
                    }
                )
        return "(totoolon;byonisevidence,hasevidence)"

    # ---- LLM ------------------------------------------------------
    def _offline_answer(self, question: str) -> str:
        """Offline answer."""
        samples = self.session.list_samples(5)["samples"]
        idx = samples[0]["index"] if samples else 0
        if self.role == "auditor":
            return json.dumps(self.session.audit(idx), ensure_ascii=False, indent=2)
        if self.role == "fixer":
            audit = self.session.audit(idx)
            suggestions = [
                f"- {f['message']}\n recommend:verifythefeatureandstable; intervene tool'thefeature'predictiondistribution;fromtrainingconcentratedremoveretraincontrast."
                for f in audit["findings"]
                if f["code"] != "none"
            ]
            return "[repairrecommend( LLM,)]\n" + ("\n".join(suggestions) or "- findingproblem,repair.")
        return self.session.offline_report(idx)
