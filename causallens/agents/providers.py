"""LLM : OpenAI ,model.

configuration(MIC_LLM_*);.
all OpenAI chat.completions ( tools/tool_calls).
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    """ .env loading(KEY=VALUE,# );settingnot."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except OSError:
        pass


_load_dotenv()


@dataclass
class Provider:
    name: str #
    base_url: str
    default_model: str
    api_key_envs: tuple[str, ...]
    note: str = ""


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        name="zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        api_key_envs=("ZHIPUAI_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY"),
        note=" GLM(glm-4-flash ;glm-4.5/4.6 supporttool)",
    ),
    Provider(
        name="zai",
        base_url="https://api.z.ai/api/paas/v4",
        default_model="glm-4.6",
        api_key_envs=("ZAI_API_KEY",),
        note="Z.ai (and)",
    ),
    Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        api_key_envs=("OPENAI_API_KEY",),
    ),
    Provider(
        name="deepseek",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        api_key_envs=("DEEPSEEK_API_KEY",),
    ),
    Provider(
        name="qwen",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        api_key_envs=("DASHSCOPE_API_KEY", "DASHSCOPE_API_KEY_INTERNATIONAL"),
        note="mode",
    ),
    Provider(
        name="moonshot",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        api_key_envs=("MOONSHOT_API_KEY",),
    ),
    Provider(
        name="siliconflow",
        base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        api_key_envs=("SILICONFLOW_API_KEY",),
        note="",
    ),
    Provider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b",
        api_key_envs=(), # key;when MIC_LLM_PROVIDER=ollama
        note=" Ollama(OpenAI ,defaultmodel)",
    ),
)


def detect_provider() -> tuple[Provider | None, str | None]:
    """return (provider, api_key).

    :MIC_LLM_PROVIDER > MIC_LLM_BASE_URL/MIC_LLM_API_KEY > .
    """
    forced = os.environ.get("MIC_LLM_PROVIDER", "").strip().lower()
    if forced:
        for p in PROVIDERS:
            if p.name == forced:
                key = next((os.environ[e] for e in p.api_key_envs if os.environ.get(e)), "ollama")
                return p, key

    if os.environ.get("MIC_LLM_BASE_URL") and os.environ.get("MIC_LLM_API_KEY"):
        return None, os.environ["MIC_LLM_API_KEY"] # ,provider=None

    for p in PROVIDERS:
        for env in p.api_key_envs:
            if os.environ.get(env):
                return p, os.environ[env]
    return None, None


def env_status() -> list[dict]:
    """llm-check :everyconfiguration."""
    rows = []
    for p in PROVIDERS:
        found = next((e for e in p.api_key_envs if os.environ.get(e)), None) if p.api_key_envs else None
        rows.append(
            {
                "provider": p.name,
                "base_url": p.base_url,
                "default_model": p.default_model,
                "key_env": p.api_key_envs[0] if p.api_key_envs else "(key)",
                "configured": bool(found) or (p.name == "ollama" and not p.api_key_envs and False),
                "note": p.note,
            }
        )
    return rows
