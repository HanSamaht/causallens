"""Investigator 循环：把审查员从"调 audit 然后复述"升级为
"生成竞争假设 → 设计实验 → 贝叶斯更新 → 显著胜出才下结论"。

设计要点（源自 LLM-as-an-Investigator 思路 + 我们自己的实测教训）：

1. 假设状态放服务端（HypothesisTracker 挂在 session 上）——LLM 不能悄悄改写历史，
   每次更新进 evidence_log，可审计。
2. 贝叶斯更新用钳制似然比（LR ∈ [0.2, 5]）：LLM 只提供定性判读（支持/反对/中性），
   数学在服务端，防单条证据掀翻全局。
3. 证据阈值：后验优势比 > ODDS_THRESHOLD 才允许宣布结论；
   证据预算耗尽仍不显著 → 必须如实报告 inconclusive（不许硬选一个）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ODDS_THRESHOLD = 5.0          # 宣布结论需要的后验优势比
LR_BOUNDS = (0.2, 5.0)        # 单条证据的似然比钳制
MAX_EVIDENCE_UPDATES = 12     # 证据预算


@dataclass
class Hypothesis:
    hid: str
    statement: str
    log_odds: float = 0.0     # 相对均匀先验的 log-odds
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.hid,
            "statement": self.statement,
            "posterior": None,  # 由 tracker 统一填充
            "n_evidence": len(self.evidence),
            "evidence": self.evidence,
        }


class HypothesisTracker:
    def __init__(self):
        self.hypotheses: dict[str, Hypothesis] = {}
        self.n_updates = 0

    # ---- 形成 ----
    def form(self, statements: list[str]) -> dict:
        if not 2 <= len(statements) <= 4:
            raise ValueError("竞争假设需要 2-4 条")
        self.hypotheses = {
            f"H{i+1}": Hypothesis(hid=f"H{i+1}", statement=s.strip())
            for i, s in enumerate(statements)
        }
        return self.snapshot()

    # ---- 更新（服务端数学，LLM 只给定性判读） ----
    def update(self, updates: list[dict]) -> dict:
        for u in updates:
            hid = u.get("hypothesis_id")
            if hid not in self.hypotheses:
                raise KeyError(f"假设 {hid} 不存在；可用: {list(self.hypotheses)}")
            direction = {"supports": +1.0, "against": -1.0, "neutral": 0.0}[
                u.get("direction", "neutral")
            ]
            weight = min(max(float(u.get("weight", 1.0)), 0.2), 5.0)
            # 钳制似然比：单步 log-odds ≤ ln(5)，weight 在 [0.2, 5] 内缩放
            step = direction * math.log(LR_BOUNDS[1]) * min(weight, LR_BOUNDS[1]) / LR_BOUNDS[1]
            self.hypotheses[hid].log_odds += step
            self.hypotheses[hid].evidence.append(
                {
                    "tool": u.get("tool_used", "?"),
                    "summary": u.get("evidence_summary", "")[:300],
                    "direction": u.get("direction"),
                    "lr_step": round(step, 3),
                }
            )
            self.n_updates += 1
        return self.snapshot()

    # ---- 后验 ----
    def posteriors(self) -> dict[str, float]:
        los = {h: hyp.log_odds for h, hyp in self.hypotheses.items()}
        m = max(los.values())
        exp = {h: math.exp(v - m) for h, v in los.items()}
        z = sum(exp.values()) or 1.0
        return {h: v / z for h, v in exp.items()}

    def snapshot(self) -> dict:
        post = self.posteriors()
        out = {h: hyp.to_dict() for h, hyp in self.hypotheses.items()}
        for h in out:
            out[h]["posterior"] = round(post[h], 4)
        return {"hypotheses": out, "n_updates": self.n_updates, "budget_left": MAX_EVIDENCE_UPDATES - self.n_updates}

    # ---- 结论 ----
    def verdict(self) -> dict:
        if not self.hypotheses:
            return {"verdict": "no_hypotheses", "note": "尚未形成假设"}
        post = self.posteriors()
        ranked = sorted(post.items(), key=lambda kv: -kv[1])
        h1, p1 = ranked[0]
        h2, p2 = ranked[1] if len(ranked) > 1 else (None, 0.0)
        odds = p1 / p2 if p2 > 1e-9 else float("inf")
        leader = self.hypotheses[h1]
        if odds >= ODDS_THRESHOLD:
            return {
                "verdict": "confirmed",
                "winner": h1,
                "statement": leader.statement,
                "posterior": round(p1, 4),
                "odds_vs_runner_up": round(odds, 2),
                "evidence": leader.evidence,
            }
        return {
            "verdict": "inconclusive",
            "leader": h1,
            "leader_statement": leader.statement,
            "leader_posterior": round(p1, 4),
            "odds_vs_runner_up": round(odds, 2) if odds != float("inf") else None,
            "note": f"证据不足以区分假设（优势比 {odds:.1f} < {ODDS_THRESHOLD}）——如实报告不确定，继续实验或降低结论强度",
            "budget_exhausted": self.n_updates >= MAX_EVIDENCE_UPDATES,
        }
