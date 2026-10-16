"""OpenAI 兼容 LLM 客户端（requests 实现，零额外依赖，多供应商加固）。

支持的供应商见 providers.py（GLM/OpenAI/DeepSeek/Qwen/Moonshot/硅基流动/Ollama/自定义）。

加固点（来自各供应商实测的兼容性坑）：
  - content 为 None（工具调用轮常见）→ 归一为 ""
  - tool_calls.function.arguments 可能是 JSON 字符串或 dict → 统一解析为 dict
  - 429/5xx 自动重试（指数退避）
  - GLM 路径坑：部分客户端丢掉 /api/paas/v4 导致 404 —— 本客户端直接拼完整 URL，不受影响
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .providers import Provider, detect_provider


class LLMConfigError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        provider, key = detect_provider()
        self.provider: Optional[Provider] = provider
        self.base_url = base_url or os.environ.get("MIC_LLM_BASE_URL") or (provider.base_url if provider else None)
        self.api_key = api_key or os.environ.get("MIC_LLM_API_KEY") or key
        self.model = model or os.environ.get("MIC_LLM_MODEL") or (provider.default_model if provider else None)
        self.timeout = float(os.environ.get("MIC_LLM_TIMEOUT", "120"))

        if not (self.base_url and self.api_key and self.model):
            raise LLMConfigError(
                "未找到 LLM 配置。三种方式任选其一：\n"
                "  1) export ZHIPUAI_API_KEY=...            （智谱 GLM，自动识别 glm-4-flash）\n"
                "  2) export OPENAI_API_KEY=... / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / MOONSHOT_API_KEY ...\n"
                "  3) export MIC_LLM_BASE_URL=... MIC_LLM_API_KEY=... [MIC_LLM_MODEL=...]  （任意 OpenAI 兼容端点）\n"
                "运行 `python -m mic llm-check` 可诊断配置。"
            )

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_tool_calls(message: dict) -> list[dict]:
        """把各家的 tool_calls 归一为 [{id, name, arguments(dict)}]；无则返回 []。"""
        out: list[dict] = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            out.append({"id": tc.get("id") or "call_0", "name": fn.get("name", ""), "arguments": args or {}})
        return out

    # ------------------------------------------------------------------
    def chat(self, messages: list[dict], tools: Optional[list[dict]] = None, temperature: float = 0.2,
             max_retries: int = 2) -> dict:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.timeout,
                )
                if resp.status_code == 401:
                    raise LLMConfigError(f"401 认证失败：API key 无效（{url}）")
                if resp.status_code == 404:
                    raise LLMConfigError(
                        f"404：路径或模型名不对。base_url={self.base_url} model={self.model}。"
                        f"GLM 完整路径须为 https://open.bigmodel.cn/api/paas/v4"
                    )
                if resp.status_code in (429, 500, 502, 503):
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return resp.json()
            except (LLMConfigError,):
                raise
            except Exception as e:  # 网络/限流：退避重试
                last_err = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 请求失败（已重试 {max_retries} 次）: {last_err}")
