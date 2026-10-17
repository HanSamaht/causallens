"""concept pack:LLM + fallback.

'partialmodelall'finally:onlymodel + background,
concept pack LLM fromfeature(GLM 50/50 passed), key asfeatureconcept.
"""
from __future__ import annotations

import json


def draft_concept_pack(feature_names: list[str], llm=None, domain_hint: str = "") -> dict:
    """returnconcept pack dict(direct ConceptPack.from_dict / json.dump)."""
    if llm is not None:
        try:
            return _draft_with_llm(feature_names, llm, domain_hint)
        except Exception:
            pass
    return {
        "domain": domain_hint or "generic",
        "name": f"concept pack( LLM,featureconcept)",
        "features": {f: {"concept": f, "description": "", "direction": "?"} for f in feature_names},
    }


def _draft_with_llm(feature_names: list[str], llm, domain_hint: str) -> dict:
    prompt = f"""is{domain_hint or ''}domain.asbelowmodelfeatureChineseconceptgrounding.
output JSON(not markdown,not):
{{"domain": "{domain_hint or 'generic'}", "name": "concept pack(LLM)",
 "features": {{feature: {{"concept": "Chinese", "description": "", "direction": "+/-/?"}}}}}}

feature:
{json.dumps(list(feature_names), ensure_ascii=False)}
"""
    resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
    text = resp["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    data = json.loads(text)
    missing = set(feature_names) - set(data["features"].keys())
    for f in missing: # LLM grounding, 100%
        data["features"][f] = {"concept": f, "description": "", "direction": "?"}
    return data
