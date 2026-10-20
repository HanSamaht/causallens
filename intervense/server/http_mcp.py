"""MCP Streamable HTTP Server( mcp SDK, v1 FastMCP / v2 MCPServer).

    python -m mic serve --demo finance --concepts concepts/finance.json \
           --http --port 8765

then MCP by URL http://127.0.0.1:8765/mcp .
stdio modeisdefault;HTTP /model.
"""
from __future__ import annotations

import json

from ..core.session import MicSession


def serve_http(session: MicSession, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve http."""
    try:
        from mcp.server.fastmcp import FastMCP # mcp v1.x
    except ImportError:
        try:
            from mcp.server.mcpserver import MCPServer as FastMCP # mcp v2.x(FastMCP )
        except ImportError as e:
            raise ImportError("HTTP require MCP SDK:pip install mcp") from e

    from ..agents.interrogator import Interrogator

    mcp = FastMCP("mic")
    interrogator = Interrogator(session)

    for schema in interrogator.tool_schemas():
        fn = schema["function"]
        tool_name, desc = fn["name"], fn.get("description", "")

        def _handler(tool_name=tool_name, **kwargs):
            """Handler."""
            return json.dumps(
                interrogator._execute_tool(tool_name, kwargs), ensure_ascii=False, default=str
            )

        # :MCP SDK astool(docstring as)
        _handler.__name__ = tool_name
        _handler.__doc__ = desc
        mcp.tool()(_handler)

    print(f"[mic] MCP HTTP server: http://{host}:{port}/mcp (model: {session.driver.name}, tool)")
    try:
        mcp.run(transport="streamable-http", host=host, port=port)
    except TypeError: # SDK run() not host/port
        mcp.settings.host, mcp.settings.port = host, port
        mcp.run(transport="streamable-http")
