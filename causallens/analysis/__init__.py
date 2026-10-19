from .attribution import explain_sample, shap_matrix
from .causal_graph import build_causal_graph
from .counterfactual import default_threshold, search_counterfactual, search_counterfactuals
from .discovery import discovery_edges
from .faithfulness import aggregate_faithfulness, evaluate_faithfulness, explain_simulate_score
from .identifiability import identify_real_effect
from .specification_audit import specification_audit

__all__ = [
    "explain_sample",
    "shap_matrix",
    "build_causal_graph",
    "discovery_edges",
    "search_counterfactual",
    "search_counterfactuals",
    "default_threshold",
    "evaluate_faithfulness",
    "aggregate_faithfulness",
    "explain_simulate_score",
    "identify_real_effect",
    "specification_audit",
]
