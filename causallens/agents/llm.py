"""OpenAI compatible LLM client over plain requests.

Provider defaults live in providers.py (GLM, OpenAI, DeepSeek, Qwen,
Moonshot, Ollama).

Normalization handled here:
  - content None on tool-call turns becomes an empty string
  - tool_calls.function.arguments arrives as JSON text or dict, becomes dict
  - 429 and 5xx responses are retried with backoff
  - GLM partial paths such as /api/paas/v4 can 404, so the full base URL
    is used directly
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .providers import Provider, detect_provider


class LLMConfigError(RuntimeError):
    """Llmconfigerror."""
    pass


class LLMClient:
    """Llmclient."""
    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        """Init."""
        provider, key = detect_provider()
        self.provider: Optional[Provider] = provider
        self.base_url = base_url or os.environ.get("MIC_LLM_BASE_URL") or (provider.base_url if provider else None)
        self.api_key = api_key or os.environ.get("MIC_LLM_API_KEY") or key
        self.model = model or os.environ.get("MIC_LLM_MODEL") or (provider.default_model if provider else None)
        self.timeout = float(os.environ.get("MIC_LLM_TIMEOUT", "120"))

        if not (self.base_url and self.api_key and self.model):
            raise LLMConfigError(
                "to LLM configuration.its:\n"
                " 1) export ZHIPUAI_API_KEY=... ( GLM,identify glm-4-flash)\n"
                " 2) export OPENAI_API_KEY=... / DEEPSEEK_API_KEY / DASHSCOPE_API_KEY / MOONSHOT_API_KEY ...\n"
                " 3) export MIC_LLM_BASE_URL=... MIC_LLM_API_KEY=... [MIC_LLM_MODEL=...] ( OpenAI )\n"
                "run `python -m mic llm-check` diagnosisconfiguration."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def normalize_tool_calls(message: dict) -> list[dict]:
        """ tool_calls as [{id, name, arguments(dict)}];return []."""
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
        """Chat."""
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
                    raise LLMConfigError(f"401 :API key invalid({url})")
                if resp.status_code == 404:
                    raise LLMConfigError(
                        f"404:ormodelnot.base_url={self.base_url} model={self.model}."
                        f"GLM as https://open.bigmodel.cn/api/paas/v4"
                    )
                if resp.status_code in (429, 500, 502, 503):
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                return resp.json()
            except (LLMConfigError,):
                raise
            except Exception as e: # /:
                last_err = e
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM ( {max_retries} ): {last_err}")
