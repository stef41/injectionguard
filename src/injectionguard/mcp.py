"""MCP (Model Context Protocol) server for real-time tool output scanning."""

from __future__ import annotations

import json
import sys
from typing import Any

from injectionguard.detector import Detector, detect, is_safe
from injectionguard.types import ThreatLevel


TOOLS = [
    {
        "name": "injectionguard_scan",
        "description": "Scan text for prompt injection patterns",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to scan for prompt injection"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "injectionguard_scan_mcp",
        "description": "Scan MCP tool output for prompt injection attacks",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool_name": {"type": "string", "description": "Name of the MCP tool that produced the output"},
                "output": {"type": "string", "description": "The tool output to scan"},
            },
            "required": ["output"],
        },
    },
    {
        "name": "injectionguard_is_safe",
        "description": "Quick boolean safety check for text",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to check"},
            },
            "required": ["text"],
        },
    },
]


def _detections_to_dicts(result) -> list[dict[str, Any]]:
    return [
        {
            "strategy": d.strategy,
            "pattern": d.pattern,
            "threat_level": d.threat_level.value,
            "message": d.message,
            "offset": d.offset,
        }
        for d in result.detections
    ]


def _error_response(id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def _success_response(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


class MCPServer:
    """MCP server that exposes injectionguard tools via JSON-RPC."""

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle a single JSON-RPC request and return a response dict."""
        req_id = request.get("id")
        method = request.get("method", "")

        if method == "initialize":
            return _success_response(req_id, {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "injectionguard", "version": "0.1.0"},
            })

        if method == "tools/list":
            return _success_response(req_id, {"tools": TOOLS})

        if method == "tools/call":
            return self._handle_tool_call(request)

        return _error_response(req_id, -32601, f"Method not found: {method}")

    def _handle_tool_call(self, request: dict[str, Any]) -> dict[str, Any]:
        req_id = request.get("id")
        params = request.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        try:
            if tool_name == "injectionguard_scan":
                return self._tool_scan(req_id, arguments)
            elif tool_name == "injectionguard_scan_mcp":
                return self._tool_scan_mcp(req_id, arguments)
            elif tool_name == "injectionguard_is_safe":
                return self._tool_is_safe(req_id, arguments)
            else:
                return _error_response(req_id, -32602, f"Unknown tool: {tool_name}")
        except Exception as exc:
            return _success_response(req_id, {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            })

    def _tool_scan(self, req_id: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        text = arguments.get("text", "")
        result = detect(text)
        result_data = {
            "is_safe": result.is_safe,
            "threat_level": result.threat_level.value,
            "detection_count": len(result.detections),
            "detections": _detections_to_dicts(result),
        }
        return _success_response(req_id, {
            "content": [{"type": "text", "text": json.dumps(result_data)}],
        })

    def _tool_scan_mcp(self, req_id: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        output = arguments.get("output", "")
        tool_name = arguments.get("tool_name", "unknown")
        result = detect(output)
        result_data = {
            "tool_name": tool_name,
            "is_safe": result.is_safe,
            "threat_level": result.threat_level.value,
            "detection_count": len(result.detections),
            "detections": _detections_to_dicts(result),
        }
        return _success_response(req_id, {
            "content": [{"type": "text", "text": json.dumps(result_data)}],
        })

    def _tool_is_safe(self, req_id: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        text = arguments.get("text", "")
        safe = is_safe(text)
        result_data = {"is_safe": safe}
        return _success_response(req_id, {
            "content": [{"type": "text", "text": json.dumps(result_data)}],
        })


def run_server() -> None:
    """Run the MCP server, reading JSON-RPC requests from stdin line by line."""
    server = MCPServer()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = _error_response(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
            continue

        response = server.handle_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    run_server()
