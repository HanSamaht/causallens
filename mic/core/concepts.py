"""概念包（Concept Pack）：领域适配的关键。

把模型的原始特征名接地为人类概念（动量 / 平均半径 / 高程……），
并可为特征标注角色（spurious=可疑, proxy=代理……）供审查员使用。
没有概念包时自动退化为"特征名即概念"，框架依然可用。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConceptEntry:
    concept: str                 # 人类概念（中文或英文）
    description: str = ""
    direction: str = ""          # 领域先验方向："+", "-", "?"（仅供提示，不作结论）
    role: str = "normal"         # normal / spurious / proxy / protected ...


@dataclass
class ConceptPack:
    domain: str = "generic"
    name: str = ""
    features: dict[str, ConceptEntry] = field(default_factory=dict)

    def ground(self, feature: str) -> Optional[ConceptEntry]:
        return self.features.get(feature)

    def concept_of(self, feature: str) -> str:
        e = self.features.get(feature)
        return e.concept if e else feature

    @classmethod
    def load(cls, path: str) -> "ConceptPack":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        feats = {
            k: ConceptEntry(
                concept=v.get("concept", k),
                description=v.get("description", ""),
                direction=v.get("direction", "?"),
                role=v.get("role", "normal"),
            )
            for k, v in data.get("features", {}).items()
        }
        return cls(domain=data.get("domain", "generic"), name=data.get("name", path), features=feats)

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptPack":
        feats = {
            k: ConceptEntry(
                concept=v.get("concept", k),
                description=v.get("description", ""),
                direction=v.get("direction", "?"),
                role=v.get("role", "normal"),
            )
            for k, v in data.get("features", {}).items()
        }
        return cls(domain=data.get("domain", "generic"), name=data.get("name", ""), features=feats)
