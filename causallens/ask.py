"""`mic ask` —— causal reasoningengine.

    python -m mic ask --demo geo --concepts concepts/geo.json \
        "ifincreases50%,mostsample?"

locate:auditis"generatereport",reasoningis"problem".
: what-if -> do intervention(//percentile)-> ->
.

honesty boundaryislayerpartial:"thisistheininterventionbelowbehavior
(modelreasoning),notprediction".
"""
from __future__ import annotations

import difflib
import json
import re

import numpy as np


class AskEngine:
    def __init__(self, session, llm=None):
        self.session = session
        self.llm = llm

    # ------------------------------------------------------------------
    def ask(self, question: str) -> str:
        drv = self.session.driver
        plan = self._parse_with_llm(question) if self.llm else None
        if plan is None:
            plan = self._parse_deterministic(question)

        # samplebyas(LLM onlyin index:N valid)
        selector = plan.get("sample_selector")
        if "mosttypical" in question or "inetc" in question:
            selector = "median"
        elif "mostsafe" in question or "lowest" in question:
            selector = "extreme_low"
        elif "mostdangerous" in question or "highest" in question or "most" in question:
            selector = "extreme_high"
        idx = self._resolve_sample(selector)
        x = drv.sample_at(idx)
        assignments = self._resolve_assignments(
            self._filter_mentioned(plan["assignments"], question), x
        )
        if not assignments: # asfallback(alsonot)
            assignments = self._resolve_assignments(plan["assignments"], x)

        before = drv.predict_one(x)
        r = self.session.intervene(idx, assignments)
        after = r["prediction_after"]

        # reasoningonbelow:thefeatureincausal role
        attr = self.session.get_attribution(idx)
        involved = {}
        for a in attr["attributions"]:
            base = a["feature"].split("@")[0]
            if base in assignments and base not in involved:
                involved[base] = a

        lines = []
        sem = "probability" if drv.prediction_semantics == "probability" else ("PF" if drv.kind == "backtest" else "score")
        lines.append(f"[Intervention]Sample #{idx}:" + ",".join(
            f"do({self.session.pack.concept_of(f.split('@')[0])} := {v:.4g})" for f, v in assignments.items()
        ))
        for f, v in assignments.items():
            old = float(x[drv.feature_names.index(f)])
            if abs(v - old) > 1e-12:
                lines.append(f" {f}: {old:.4g} -> {v:.4g} ({(v / old - 1) if abs(old) > 1e-12 else float('inf'):+.0%})")
        lines.append(f"[Answer]the{sem}: {before:.4g} -> {after:.4g}({after - before:+.4g})")
        for base, a in involved.items():
            lines.append(
                f"[Causal Role]{a['concept']}({base}) intheattribution {a['share']:.0%}"
                + (" ⚠️suspicious feature" if a["role"] == "spurious" else "")
            )
        boundary = "This is in-model reasoning, not real-world prediction."

        if getattr(drv, "kind", "") == "decision-log":
            boundary += (
                f" predict ,reasoningmodel:consistent "
                f"{drv.fidelity['action_agreement']:.1%}(conclusionon)."
            )
        lines.append("[Boundary]" + boundary)
        return "\n".join(lines)

    def _filter_mentioned(self, assignments: list[dict], question: str) -> list[dict]:
        """:LLM onlyinterventiontofeature——range."""
        pack = self.session.pack
        kept = []
        for a in assignments:
            f = a.get("feature", "")
            base = f.split("@")[0]
            entry = pack.ground(base)
            names = {base, f, entry.concept if entry else ""}
            if any(n and n in question for n in names):
                kept.append(a)
        return kept

    # ------------------------------------------------------------------
    # sample:index:N / extreme_high / extreme_low / median
    def _resolve_sample(self, selector) -> int:
        drv = self.session.driver
        s = str(selector or "extreme_high")
        p = drv.predict(drv.background[:500])
        if s.startswith("index:") and s.split(":", 1)[1].isdigit():
            return int(s.split(":", 1)[1])
        if s == "extreme_low":
            return int(np.argmin(p))
        if s == "median":
            return int(np.argsort(np.abs(p - np.median(p)))[0])
        return int(np.argmax(p)) # extreme_high default:most""samplemosthasreasoningvalue

    # value:LLM only"",inthis()
    def _resolve_assignments(self, raw: list[dict], x: np.ndarray) -> dict[str, float]:
        drv = self.session.driver
        out = {}
        for a in raw:
            f = a["feature"]
            if f not in drv.feature_names:
                f = self._match_feature(f)
                if f is None:
                    continue
            old = float(x[drv.feature_names.index(f)])
            op = a.get("op", "set")
            if op == "scale":
                out[f] = old * float(a["value"])
            elif op == "quantile":
                out[f] = drv.quantile_of(f, float(a["value"]) / 100.0)
            else:
                out[f] = float(a["value"])
        return out

    def _match_feature(self, text: str) -> str | None:
        names = self.session.driver.feature_names
        bases = {n.split("@")[0] for n in names}
        pack = self.session.pack
        concept_map = {pack.ground(b).concept if pack.ground(b) else b: b for b in bases}
        for key, base in concept_map.items():
            if text in key or key in text:
                return next(n for n in names if n.split("@")[0] == base and "@" not in n) or \
                    next(n for n in names if n.split("@")[0] == base)
        m = difflib.get_close_matches(text, list(concept_map.keys()), n=1, cutoff=0.4)
        if m:
            base = concept_map[m[0]]
            return next((n for n in names if n.split("@")[0] == base and "@" not in n),
                        next(n for n in names if n.split("@")[0] == base))
        return None

    # ------------------------------------------------------------------
    def _parse_with_llm(self, question: str) -> dict | None:
        pack = self.session.pack
        vocab = {
            f: (pack.ground(f.split("@")[0]).concept if pack.ground(f.split("@")[0]) else f)
            for f in self.session.driver.feature_names
        }
        prompt = f""" what-if problemintervention.feature(:concept):{json.dumps(vocab, ensure_ascii=False)}
output JSON(not):
{{"assignments": [{{"feature": "feature(muston)", "op": "set|scale|quantile",
   "value": value(set=value; scale=1.5+50%; quantile=percentile0-100)}}],
 "sample_selector": "index:N | extreme_high | extreme_low | median"}}
problem:{question}
:"increasesX%" scale: 1+X/100;"rises tohistory90percentile" quantile: 90;"set to/becomesvalue" set."""
        try:
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
            text = resp["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.split("```")[1].lstrip("json").strip()
            plan = json.loads(text)
            if plan.get("assignments"):
                return plan
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    _PAT_SET = re.compile(r"if\s+(.{1,20}?)\s+(?:is set to|becomes|equals|to)\s+([+-]?[0-9.]+)", re.I)
    _PAT_SCALE = re.compile(r"if\s+(.{1,20}?)\s+(?:increases?|decreases?|rises?|drops?)\s+(?:by\s+)?([0-9.]+)\s*%", re.I)
    _PAT_QUANT = re.compile(r"if\s+(.{1,20}?)\s+(?:rises?|drops?|goes?)\s+to\s+(?:the\s+)?([0-9.]+)(?:th)?\s*(?:percentile|quantile)", re.I)
    _PAT_SAMPLE = re.compile(r"sample\s+(\d+)", re.I)

    def _parse_deterministic(self, question: str) -> dict:
        assignments = []
        m = self._PAT_SCALE.search(question)
        if m:
            mult = float(m.group(2)) / 100.0
            if any(w in question.lower() for w in ("decrease", "drop", "down")):
                mult = -mult
            assignments.append({"feature": self._match_feature(m.group(1)) or m.group(1),
                                "op": "scale", "value": 1 + mult})
        else:
            mq = self._PAT_QUANT.search(question)
            m2 = self._PAT_SET.search(question)
            if mq:
                assignments.append({"feature": self._match_feature(mq.group(1)) or mq.group(1),
                                    "op": "quantile", "value": float(mq.group(2))})
            elif m2:
                assignments.append({"feature": self._match_feature(m2.group(1)) or m2.group(1),
                                    "op": "set", "value": float(m2.group(2))})
        if not assignments:
            raise ValueError(f"Cannot parse what-if question: {question}")
        sel = "extreme_high"
        ms = self._PAT_SAMPLE.search(question)
        if ms:
            sel = f"index:{ms.group(1)}"
        elif "lowest" in question.lower() or "safest" in question.lower():
            sel = "extreme_low"
        elif "typical" in question.lower() or "median" in question.lower():
            sel = "median"
        return {"assignments": assignments, "sample_selector": sel}
