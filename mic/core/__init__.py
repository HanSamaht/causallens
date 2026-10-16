from .concepts import ConceptPack
from .types import (
    Attribution,
    CausalEdge,
    CausalGraph,
    CounterfactualResult,
    FaithfulnessReport,
)


def __getattr__(name: str):
    # MicSession 依赖 analysis 包；惰性导出以打破 core ↔ analysis 的循环导入
    if name == "MicSession":
        from .session import MicSession

        return MicSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
