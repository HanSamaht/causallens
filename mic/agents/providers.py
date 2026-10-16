"""LLM 供应商注册表：一套 OpenAI 兼容协议，多家主流模型即插即用。

显式配置（MIC_LLM_*）永远优先；否则按优先级自动探测环境变量。
所有供应商均已官方兼容 OpenAI chat.completions 协议（含 tools/tool_calls）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _load_dotenv(path: str = ".env") -> None:
    """极简 .env 加载（KEY=VALUE，# 注释）；已设置的环境变量不覆盖。"""
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
    name: str            # 供应商标识
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
        note="智谱 GLM（glm-4-flash 免费；glm-4.5/4.6 支持工具调用）",
    ),
    Provider(
        name="zai",
        base_url="https://api.z.ai/api/paas/v4",
        default_model="glm-4.6",
        api_key_envs=("ZAI_API_KEY",),
        note="Z.ai 国际端点（与智谱国内平台同协议）",
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
        note="阿里通义千问兼容模式",
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
        note="硅基流动聚合平台",
    ),
    Provider(
        name="ollama",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b",
        api_key_envs=(),  # 本地无需 key；仅当显式 MIC_LLM_PROVIDER=ollama 时启用
        note="本地 Ollama（OpenAI 兼容端点，默认模型请按本地已拉取的填写）",
    ),
)


def detect_provider() -> tuple[Provider | None, str | None]:
    """返回 (provider, api_key)。

    优先级：MIC_LLM_PROVIDER 指定 > MIC_LLM_BASE_URL/MIC_LLM_API_KEY 显式 > 环境变量探测。
    """
    forced = os.environ.get("MIC_LLM_PROVIDER", "").strip().lower()
    if forced:
        for p in PROVIDERS:
            if p.name == forced:
                key = next((os.environ[e] for e in p.api_key_envs if os.environ.get(e)), "ollama")
                return p, key

    if os.environ.get("MIC_LLM_BASE_URL") and os.environ.get("MIC_LLM_API_KEY"):
        return None, os.environ["MIC_LLM_API_KEY"]  # 自定义端点，provider=None

    for p in PROVIDERS:
        for env in p.api_key_envs:
            if os.environ.get(env):
                return p, os.environ[env]
    return None, None


def env_status() -> list[dict]:
    """llm-check 用：每个供应商的环境变量配置状态。"""
    rows = []
    for p in PROVIDERS:
        found = next((e for e in p.api_key_envs if os.environ.get(e)), None) if p.api_key_envs else None
        rows.append(
            {
                "provider": p.name,
                "base_url": p.base_url,
                "default_model": p.default_model,
                "key_env": p.api_key_envs[0] if p.api_key_envs else "(免key)",
                "configured": bool(found) or (p.name == "ollama" and not p.api_key_envs and False),
                "note": p.note,
            }
        )
    return rows
