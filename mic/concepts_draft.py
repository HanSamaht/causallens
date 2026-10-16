"""概念包半自动起草：LLM 起草 + 失败回退通用包。

目标『让大部分黑箱模型都能说话』的最后一公里：用户只需模型 + 背景数据，
概念包由 LLM 从特征名起草（GLM 实测 50/50 通过），无 key 时退化为特征名即概念。
"""
from __future__ import annotations

import json


def draft_concept_pack(feature_names: list[str], llm=None, domain_hint: str = "") -> dict:
    """返回概念包 dict（可直接 ConceptPack.from_dict / json.dump）。"""
    if llm is not None:
        try:
            return _draft_with_llm(feature_names, llm, domain_hint)
        except Exception:
            pass
    return {
        "domain": domain_hint or "generic",
        "name": f"自动概念包（无 LLM，特征名即概念）",
        "features": {f: {"concept": f, "description": "", "direction": "?"} for f in feature_names},
    }


def _draft_with_llm(feature_names: list[str], llm, domain_hint: str) -> dict:
    prompt = f"""你是{domain_hint or '机器学习'}领域专家。为下列模型特征给出中文概念接地。
输出严格 JSON（不要 markdown、不要多余文字）：
{{"domain": "{domain_hint or 'generic'}", "name": "概念包(LLM起草)",
 "features": {{特征名: {{"concept": "简短中文名", "description": "一句话解释", "direction": "+/-/?"}}}}}}

特征名：
{json.dumps(list(feature_names), ensure_ascii=False)}
"""
    resp = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
    text = resp["choices"][0]["message"]["content"].strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    data = json.loads(text)
    missing = set(feature_names) - set(data["features"].keys())
    for f in missing:  # LLM 漏的补通用接地，保证 100% 覆盖
        data["features"][f] = {"concept": f, "description": "", "direction": "?"}
    return data
