from .concepts import ConceptPack
from .types import (
    Attribution,
    CausalEdge,
    CausalGraph,
    CounterfactualResult,
    FaithfulnessReport,
)


def __getattr__(name: str):
    """Lazy import to avoid circular dependency."""
    # MicSession analysis ;by core ↔ analysis
    if name == "MicSession":
        from .session import MicSession

        return MicSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
