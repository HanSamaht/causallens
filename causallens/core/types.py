"""in(Evidence IR):analysisallisevidenceobject.

LLM layeronlyallowthisevidence——thisis.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Attribution:
    """featurepredictionattribution(SHAP value)."""

    feature: str
    shap_value: float
    feature_value: float
    concept: Optional[str] = None # concept packgroundingconcept( "")
    role: Optional[str] = None # concept packtaggedrole(normal / spurious / proxy ...)

    @property
    def signed_pct(self) -> str:
        return f"{self.shap_value:+.3f}"


@dataclass
class CounterfactualResult:
    """counterfactual:mostprediction."""

    base_prediction: float
    cf_prediction: float
    changed_features: dict[str, tuple[float, float]] # feature -> (value, value)
    distance: float # L1
    flipped: bool
    method: str = "quantile_sweep"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class CausalEdge:
    """causal graph.

    kind:
      - "direct": feature -> prediction effect( SHAP )
      - "interaction": feature f -> feature g attribution interventioneffect( f set tonotpercentilevalue,
                          g attribution——thisisintervention experiment,notcorrelation)
      - "correlational": observational only,,notandcausalconclusion
    """

    source: str
    target: str
    effect: float
    kind: str = "direct"
    detail: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class CausalGraph:
    """interrogationtocausal graph(lite :inputfeaturelayer + prediction)."""

    nodes: list[str] = field(default_factory=list)
    edges: list[CausalEdge] = field(default_factory=list)
    method: str = "shap_aggregation + quantile_intervention"

    @property
    def direct_edges(self) -> list[CausalEdge]:
        return [e for e in self.edges if e.kind == "direct"]

    @property
    def interaction_edges(self) -> list[CausalEdge]:
        return [e for e in self.edges if e.kind == "interaction"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "method": self.method,
        }


@dataclass
class FaithfulnessReport:
    """faithfulness verification:attributionismodel.

    comprehensiveness(comprehensiveness): Top-k featurebackgroundvalue,prediction;
    sufficiency(sufficiency): only Top-k,itsvalue,predictionunchanged;
    random_baseline: ,SHAP significantmore.
    """

    top_k: int
    comprehensiveness: float # top-k prediction
    sufficiency: float # only top-k prediction
    random_baseline: float # top-k prediction
    passed: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
