"""Bonus: expose the audited tool registry as a real MCP server.

The `mcp` python package installed cleanly in this environment, so the
registry can be served over stdio to any MCP client:

    .venv/bin/python -m sensorflow.retro.tools.mcp_server

Design notes:
- Every MCP tool call still flows through ToolRegistry.call(), so the audit
  trail, schema validation, timeouts, and write-authorization gate apply
  identically to MCP clients and in-process callers.
- The write tool (create_evaluation_case) is exposed but the registry will
  deny it unless the client passes policy_authorization=true in arguments —
  the MCP layer cannot bypass the boundary.
- If `mcp` is not installed this module raises a clear ImportError; the
  registry itself remains the MCP-style boundary (documented in registry.py).
"""

from __future__ import annotations

from typing import Any, Dict

from sensorflow.retro.tools.builtin import build_registry


def _import_server_class():
    """Support both the modern (MCPServer) and older (FastMCP) APIs."""
    try:
        from mcp.server.mcpserver import MCPServer
        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP
        return FastMCP
    except ImportError as exc:  # honest failure; registry remains the boundary
        raise ImportError(
            "the 'mcp' package is not installed (or has an unknown server "
            "API); the ToolRegistry remains the MCP-style boundary "
            "(pip install mcp to enable this server)") from exc


def build_mcp_server():
    """Construct (but do not run) the MCP server wrapping the registry."""
    server_cls = _import_server_class()
    registry = build_registry(analysis_id="mcp-session")
    server = server_cls(name="sensorflow-retro")

    def make_handler(tool_name: str):
        def handler(arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
            args = dict(arguments or {})
            authorized = bool(args.pop("policy_authorization", False))
            res = registry.call(tool_name, args, policy_authorization=authorized)
            return res.model_dump()
        return handler

    for spec in registry.specs():
        server.add_tool(
            make_handler(spec.name),
            name=spec.name,
            description=(f"{spec.description} "
                         f"[read_only={spec.read_only}, timeout={spec.timeout_s}s; "
                         f"args: single 'arguments' object matching the input "
                         f"schema; write tools need policy_authorization=true]"))
    return server, registry


if __name__ == "__main__":
    srv, _ = build_mcp_server()
    srv.run()  # stdio transport
