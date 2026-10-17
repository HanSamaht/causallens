"""Investigator :auditorfrom" audit then"upgradeas
"generatehypothesis -> -> update -> significantwinnerbelowconclusion".

( LLM-as-an-Investigator + ):

1. hypothesis(HypothesisTracker in session on)——LLM nothistory,
   everyupdate evidence_log,audit.
2. update(LR ∈ [0.2, 5]):LLM only(support//in),
   in,evidence.
3. evidencethreshold:posterior > ODDS_THRESHOLD allowconclusion;
   evidencenotsignificant -> mustreport inconclusive(not).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

ODDS_THRESHOLD = 5.0 # conclusionrequireposterior
LR_BOUNDS = (0.2, 5.0) # evidence
MAX_EVIDENCE_UPDATES = 12 # evidence


@dataclass
class Hypothesis:
    """Hypothesis."""
    hid: str
    statement: str
    log_odds: float = 0.0 # prior log-odds
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """To dict."""
        return {
            "id": self.hid,
            "statement": self.statement,
            "posterior": None, # tracker
            "n_evidence": len(self.evidence),
            "evidence": self.evidence,
        }


class HypothesisTracker:
    """Hypothesistracker."""
    def __init__(self):
        """Init."""
        self.hypotheses: dict[str, Hypothesis] = {}
        self.n_updates = 0

    # ---- ----
    def form(self, statements: list[str]) -> dict:
        """Form."""
        if not 2 <= len(statements) <= 4:
            raise ValueError("hypothesisrequire 2-4 ")
        self.hypotheses = {
            f"H{i+1}": Hypothesis(hid=f"H{i+1}", statement=s.strip())
            for i, s in enumerate(statements)
        }
        return self.snapshot()

    # ---- update(,LLM only) ----
    def update(self, updates: list[dict]) -> dict:
        """Update."""
        for u in updates:
            hid = u.get("hypothesis_id")
            if hid not in self.hypotheses:
                raise KeyError(f"hypothesis {hid} notin;: {list(self.hypotheses)}")
            direction = {"supports": +1.0, "against": -1.0, "neutral": 0.0}[
                u.get("direction", "neutral")
            ]
            weight = min(max(float(u.get("weight", 1.0)), 0.2), 5.0)
            # : log-odds ≤ ln(5),weight in [0.2, 5]
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

    # ---- posterior ----
    def posteriors(self) -> dict[str, float]:
        """Posteriors."""
        los = {h: hyp.log_odds for h, hyp in self.hypotheses.items()}
        m = max(los.values())
        exp = {h: math.exp(v - m) for h, v in los.items()}
        z = sum(exp.values()) or 1.0
        return {h: v / z for h, v in exp.items()}

    def snapshot(self) -> dict:
        """Snapshot."""
        post = self.posteriors()
        out = {h: hyp.to_dict() for h, hyp in self.hypotheses.items()}
        for h in out:
            out[h]["posterior"] = round(post[h], 4)
        return {"hypotheses": out, "n_updates": self.n_updates, "budget_left": MAX_EVIDENCE_UPDATES - self.n_updates}

    # ---- conclusion ----
    def verdict(self) -> dict:
        """Verdict."""
        if not self.hypotheses:
            return {"verdict": "no_hypotheses", "note": "hypothesis"}
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
            "note": f"insufficient evidence to distinguish hypotheses( {odds:.1f} < {ODDS_THRESHOLD})——reportnot,ordecreasesconclusion",
            "budget_exhausted": self.n_updates >= MAX_EVIDENCE_UPDATES,
        }
