"""极简 MCP stdio Server（零依赖：JSON-RPC 2.0 over stdin/stdout）。

把 MicSession 的审讯工具暴露给任何 MCP 客户端（Claude Desktop / 任意 LLM 客户端），
配置示例（claude_desktop_config.json）：

    {
      "mcpServers": {
        "mic-finance": {
          "command": "python",
          "args": ["-m", "mic", "serve",
                   "--model", "examples/artifacts/finance_model.pkl",
                   "--background", "examples/artifacts/finance_background.csv",
                   "--concepts", "concepts/finance_zh.json"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys

from ..agents.interrogator import Interrogator
from ..core.session import MicSession

PROTOCOL_VERSION = "2024-11-05"


def _tool_defs(session: MicSession) -> list[dict]:
    """把 Interrogator 的 OpenAI 工具 schema 平移为 MCP tool 定义。"""
    tools = []
    for t in Interrogator(session).tool_schemas():
        fn = t["function"]
        tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return tools


def _execute(session: MicSession, name: str, args: dict) -> dict:
    return Interrogator(session)._execute_tool(name, args)


def serve_stdio(session: MicSession) -> None:
    out = sys.stdout

    def send(payload: dict) -> None:
        out.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method", "")
        req_id = req.get("id")

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "mic", "version": "0.1.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            pass  # 通知无需应答
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": req_id, "result": {"tools": _tool_defs(session)}})
        elif method == "tools/call":
            params = req.get("params", {})
            result = _execute(session, params.get("name", ""), params.get("arguments", {}) or {})
            send(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False, default=str)}],
                        "isError": "error" in result,
                    },
                }
            )
        elif req_id is not None:
            send({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"未知方法 {method}"}})
