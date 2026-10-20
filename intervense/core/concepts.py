"""concept pack(Concept Pack):domain.

modelfeaturegroundingasconcept( / / ……),
asfeaturetaggedrole(spurious=suspicious, proxy=proxy……)auditor.
noconcept packas"featureconcept",framework.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConceptEntry:
    """Conceptentry."""
    concept: str # concept(ChineseorEnglish)
    description: str = ""
    direction: str = "" # domainpriordirection:"+", "-", "?"(,notconclusion)
    role: str = "normal" # normal / spurious / proxy / protected ...


@dataclass
class ConceptPack:
    """Conceptpack."""
    domain: str = "generic"
    name: str = ""
    features: dict[str, ConceptEntry] = field(default_factory=dict)

    def ground(self, feature: str) -> Optional[ConceptEntry]:
        """Ground."""
        return self.features.get(feature)

    def concept_of(self, feature: str) -> str:
        """Concept of."""
        e = self.features.get(feature)
        return e.concept if e else feature

    @classmethod
    def load(cls, path: str) -> "ConceptPack":
        """Load."""
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
        """From dict."""
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
