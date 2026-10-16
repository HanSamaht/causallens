"""`mic ask` —— 因果推演引擎的第一入口。

    python -m mic ask --demo geo --concepts concepts/geo_zh.json \
        "如果降雨量增加50%，最容易滑坡的样本风险怎么变？"

产品定位转变的落点：审计的入口是"生成报告"，推演的入口是"问一个问题"。
流程：解析 what-if → 服务端解析成 do 干预（相对/绝对/分位）→ 跑实验 →
带忠实性边界的回答。

诚实边界是产品层的一部分：回答永远声明"这是该系统在干预下的行为变化
（模型内推演），不是现实世界预测"。
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

        # 样本选择的常用说法以服务端裁决为准（LLM 只在 index:N 时有效）
        selector = plan.get("sample_selector")
        if "最典型" in question or "中等" in question:
            selector = "median"
        elif "最安全" in question or "最低" in question:
            selector = "extreme_low"
        elif "最危险" in question or "最高" in question or "最容易" in question:
            selector = "extreme_high"
        idx = self._resolve_sample(selector)
        x = drv.sample_at(idx)
        assignments = self._resolve_assignments(
            self._filter_mentioned(plan["assignments"], question), x
        )
        if not assignments:  # 过滤后为空则回退原始解析（宁可用原案也不空手）
            assignments = self._resolve_assignments(plan["assignments"], x)

        before = drv.predict_one(x)
        r = self.session.intervene(idx, assignments)
        after = r["prediction_after"]

        # 推演上下文：该特征在系统里的因果角色
        attr = self.session.get_attribution(idx)
        involved = {}
        for a in attr["attributions"]:
            base = a["feature"].split("@")[0]
            if base in assignments and base not in involved:
                involved[base] = a

        lines = []
        sem = "概率" if drv.prediction_semantics == "probability" else ("PF" if drv.kind == "backtest" else "分数")
        lines.append(f"【推演】样本#{idx}：" + "，".join(
            f"do({self.session.pack.concept_of(f.split('@')[0])} := {v:.4g})" for f, v in assignments.items()
        ))
        for f, v in assignments.items():
            old = float(x[drv.feature_names.index(f)])
            if abs(v - old) > 1e-12:
                lines.append(f"  {f}: {old:.4g} → {v:.4g} ({(v / old - 1) if abs(old) > 1e-12 else float('inf'):+.0%})")
        lines.append(f"【回答】该系统{sem}: {before:.4g} → {after:.4g}（{after - before:+.4g}）")
        for base, a in involved.items():
            lines.append(
                f"【因果角色】{a['concept']}({base}) 在该系统全局归因占比 {a['share']:.0%}"
                + ("  ⚠️可疑特征" if a["role"] == "spurious" else "")
            )
        boundary = "以上是该系统在此干预下的行为变化（模型内推演），不是对现实世界的预测。"
        if getattr(drv, "kind", "") == "decision-log":
            boundary += (
                f" 本系统无 predict 函数，推演基于替代模型：动作一致率 "
                f"{drv.fidelity['action_agreement']:.1%}（结论可信度上限）。"
            )
        lines.append("【边界】" + boundary)
        return "\n".join(lines)

    def _filter_mentioned(self, assignments: list[dict], question: str) -> list[dict]:
        """纪律：LLM 只能干预用户明确提到的特征——防它自作主张扩大手术范围。"""
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
    # 样本选择：index:N / extreme_high / extreme_low / median
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
        return int(np.argmax(p))  # extreme_high 默认：最"阳性"的样本最有推演价值

    # 服务端解析赋值：LLM 只说"操作"，数学在这边算（诚实纪律）
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
        prompt = f"""把用户的 what-if 问题解析成干预计划。可用特征(名:概念)：{json.dumps(vocab, ensure_ascii=False)}
输出严格 JSON（不要多余文字）：
{{"assignments": [{{"feature": "特征名(必须来自上面列表)", "op": "set|scale|quantile",
   "value": 数值(set=绝对值; scale=倍数如1.5表示+50%; quantile=百分位0-100)}}],
 "sample_selector": "index:N | extreme_high | extreme_low | median"}}
问题：{question}
注意：表达"增加X%"用 scale: 1+X/100；"升到历史90分位"用 quantile: 90；"设为/变成某值"用 set。"""
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
    _PAT_SET = re.compile(r"如果(.{1,14}?)(?:从[\d.]+)?(?:变成|变为|调整?到|设为|升到|降到|为)([+-]?[\d.]+)")
    _PAT_QUANT = re.compile(r"如果(.{1,14}?)(?:升到|降到|提到)(?:历史|过去的?)?([\d.]+)分位")
    _PAT_SCALE = re.compile(r"如果(.{1,14}?)(?:增加|提高|上升|降低|减少|下降)([\d.]+)%")
    _PAT_SAMPLE = re.compile(r"第(\d+)个样本")

    def _parse_deterministic(self, question: str) -> dict:
        assignments = []
        m = self._PAT_SCALE.search(question)
        if m:
            mult = float(m.group(2)) / 100.0
            if any(w in question for w in ("降低", "减少", "下降")):
                mult = -mult
            assignments.append({"feature": self._match_feature(m.group(1)) or m.group(1),
                                "op": "scale", "value": 1 + mult})
        else:
            mq = self._PAT_QUANT.search(question)
            m = self._PAT_SET.search(question)
            if mq:
                assignments.append({"feature": self._match_feature(mq.group(1)) or mq.group(1),
                                    "op": "quantile", "value": float(mq.group(2))})
            elif m:
                assignments.append({"feature": self._match_feature(m.group(1)) or m.group(1),
                                    "op": "set", "value": float(m.group(2))})
        if not assignments:
            raise ValueError(f"无法解析 what-if 问题：{question}（无 LLM 时请用『如果<概念>增加X%』句式）")
        sel = "extreme_high"
        ms = self._PAT_SAMPLE.search(question)
        if ms:
            sel = f"index:{ms.group(1)}"
        elif "最安全" in question or "最低" in question:
            sel = "extreme_low"
        return {"assignments": assignments, "sample_selector": sel}
