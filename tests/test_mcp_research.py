import httpx
import pytest

from orchestrator.mcp_research import McpResearcher, mcp_integrations


def test_mcp_integrations_single(tmp_path):
    f = tmp_path / "mcp.json"
    f.write_text('{"mcpServers": {"exa": {"url": "https://x"}}}', encoding="utf-8")
    assert mcp_integrations(f) == ["mcp/exa"]


def test_mcp_integrations_multiple(tmp_path):
    f = tmp_path / "mcp.json"
    f.write_text(
        '{"mcpServers": {"exa": {"url": "x"}, "playwright": {"command": "y"}}}',
        encoding="utf-8",
    )
    assert sorted(mcp_integrations(f)) == ["mcp/exa", "mcp/playwright"]


def test_mcp_integrations_missing_or_bad(tmp_path):
    assert mcp_integrations(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert mcp_integrations(bad) == []


def _researcher(handler) -> McpResearcher:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return McpResearcher(
        base_url="http://localhost:1234", token="sk-lm-test",
        model="m", integrations=["mcp/exa"], http_client=client,
    )


async def test_research_request_shape_and_parse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = __import__("json").loads(request.read())
        return httpx.Response(200, json={"output": [
            {"type": "reasoning", "content": "thinking"},
            {"type": "tool_call", "tool": "web_search_exa", "arguments": {}, "output": "..."},
            {"type": "message", "content": "the answer"},
        ]})

    r = _researcher(handler)
    out = await r.research("find news")
    assert captured["path"] == "/api/v1/chat"
    assert captured["auth"] == "Bearer sk-lm-test"
    assert captured["body"]["input"] == "find news"
    assert captured["body"]["integrations"] == ["mcp/exa"]
    assert "web_search_exa" in out
    assert "the answer" in out


async def test_research_handles_list_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": [
            {"type": "message", "content": [{"type": "output_text", "text": "part1 "},
                                            {"type": "output_text", "text": "part2"}]},
        ]})

    out = await _researcher(handler).research("q")
    assert "part1 part2" in out


async def test_research_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(httpx.HTTPStatusError):
        await _researcher(handler).research("q")
