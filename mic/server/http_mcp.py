"""MCP Streamable HTTP Server（基于官方 mcp SDK，兼容 v1 FastMCP / v2 MCPServer）。

    python -m mic serve --demo finance --concepts concepts/finance_zh.json \
           --http --port 8765

然后任何 MCP 客户端以 URL http://127.0.0.1:8765/mcp 接入。
stdio 模式仍是零依赖默认；HTTP 用于远程/多客户端共享同一模型的场景。
"""
from __future__ import annotations

import json

from ..core.session import MicSession


def serve_http(session: MicSession, host: str = "127.0.0.1", port: int = 8765) -> None:
    try:
        from mcp.server.fastmcp import FastMCP  # mcp v1.x
    except ImportError:
        try:
            from mcp.server.mcpserver import MCPServer as FastMCP  # mcp v2.x（FastMCP 改名）
        except ImportError as e:
            raise ImportError("HTTP 传输需要官方 MCP SDK：pip install mcp") from e

    from ..agents.interrogator import Interrogator

    mcp = FastMCP("mic")
    interrogator = Interrogator(session)

    for schema in interrogator.tool_schemas():
        fn = schema["function"]
        tool_name, desc = fn["name"], fn.get("description", "")

        def _handler(tool_name=tool_name, **kwargs):
            return json.dumps(
                interrogator._execute_tool(tool_name, kwargs), ensure_ascii=False, default=str
            )

        # 先定名再注册：MCP SDK 用函数名作为工具名（docstring 作为描述）
        _handler.__name__ = tool_name
        _handler.__doc__ = desc
        mcp.tool()(_handler)

    print(f"[mic] MCP HTTP server: http://{host}:{port}/mcp  (模型: {session.driver.name}, 工具已注册)")
    try:
        mcp.run(transport="streamable-http", host=host, port=port)
    except TypeError:  # 旧版 SDK 的 run() 不收 host/port
        mcp.settings.host, mcp.settings.port = host, port
        mcp.run(transport="streamable-http")
