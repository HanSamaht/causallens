"""解释中间表示（Evidence IR）：一切分析产物都是可序列化的证据对象。

LLM 层只允许引用这些证据里的数字说话——这是防幻觉的架构约束。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Attribution:
    """单个特征对单次预测的归因（SHAP 值）。"""

    feature: str
    shap_value: float
    feature_value: float
    concept: Optional[str] = None        # 概念包接地后的人类概念（如 "动量"）
    role: Optional[str] = None           # 概念包标注的角色（normal / spurious / proxy ...）

    @property
    def signed_pct(self) -> str:
        return f"{self.shap_value:+.3f}"


@dataclass
class CounterfactualResult:
    """反事实：最小改动使预测翻转。"""

    base_prediction: float
    cf_prediction: float
    changed_features: dict[str, tuple[float, float]]  # feature -> (原值, 新值)
    distance: float                                    # 归一化 L1 距离
    flipped: bool
    method: str = "quantile_sweep"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class CausalEdge:
    """因果图的一条边。

    kind:
      - "direct":        特征 → 预测 的总效应（由 SHAP 聚合）
      - "interaction":   特征 f → 特征 g 的归因 的干预效应（把 f 设为不同分位值，
                         观察 g 的归因变化——这是真正的干预实验，不是相关）
      - "correlational": 仅观测相关，供参考，不参与因果结论
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
    """一次审讯得到的因果图（lite 版：输入特征层 + 预测节点）。"""

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
    """忠实性验证：归因排名是否真的驱动模型决策。

    comprehensiveness（完备性）: 把 Top-k 特征抹回背景均值后，预测应大幅变化；
    sufficiency（充分性）:       只保留 Top-k、其余抹回均值，预测应基本不变；
    random_baseline:             随机排名做同样手术，SHAP 排名应显著更强。
    """

    top_k: int
    comprehensiveness: float          # 抹掉 top-k 后的预测变化量
    sufficiency: float                # 只留 top-k 后的预测变化量
    random_baseline: float            # 随机 top-k 抹除后的预测变化量
    passed: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
