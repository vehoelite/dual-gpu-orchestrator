"""MCP-backed research via LM Studio's native /api/v1/chat endpoint.

LM Studio runs the user's mcp.json servers (tool discovery + execution + loop)
server-side and returns the result. We write no MCP protocol code — we just call
the native endpoint with the configured server integrations. See
docs/reference/lmstudio-mcp-via-api.md."""
from __future__ import annotations

import json
from pathlib import Path

import httpx


def mcp_integrations(mcp_json_path) -> list[str]:
    """Return ['mcp/<id>', ...] for each server in mcp.json. Missing or
    unparseable file -> []."""
    p = Path(mcp_json_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    servers = data.get("mcpServers", {})
    return [f"mcp/{name}" for name in servers]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        return "".join(parts)
    return "" if content is None else str(content)


def _extract_answer(output: list) -> str:
    """Take the last `type == "message"` entry as the answer; prefix a note of
    which tools ran."""
    if not isinstance(output, list):
        return str(output)[:2000]
    messages = [o for o in output if isinstance(o, dict) and o.get("type") == "message"]
    text = _content_text(messages[-1].get("content")) if messages else ""
    tools = [
        o.get("tool")
        for o in output
        if isinstance(o, dict) and o.get("type") == "tool_call" and o.get("tool")
    ]
    note = f"[used: {', '.join(tools)}]\n" if tools else ""
    return (note + text).strip() or json.dumps(output)[:2000]


class McpResearcher:
    def __init__(
        self,
        base_url: str,
        token: str,
        model: str,
        integrations: list[str],
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model = model
        self.integrations = integrations
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def research(self, query: str) -> str:
        payload = {
            "model": self.model,
            "input": query,
            "integrations": self.integrations,
        }
        resp = await self._client.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {self.token}"},
        )
        resp.raise_for_status()
        return _extract_answer(resp.json().get("output", []))

    async def aclose(self) -> None:
        await self._client.aclose()
