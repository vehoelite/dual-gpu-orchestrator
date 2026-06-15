import httpx
import pytest

from orchestrator.planner import GeminiPlanner, LocalPlanner


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []

    async def complete(self, model, messages, temperature=0.7):
        self.calls.append((model, messages))
        return self.text


async def test_local_planner_parses_model_output():
    client = FakeClient("1. write tests\n2. implement")
    planner = LocalPlanner(client=client, model="dom")
    steps = await planner.make_plan("build a thing")
    assert steps == ["write tests", "implement"]
    assert client.calls[0][1][-1]["content"] == "build a thing"


async def test_gemini_planner_request_and_parse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.url.params.get("key")
        return httpx.Response(
            200,
            json={"candidates": [
                {"content": {"parts": [{"text": "1. step one\n2. step two"}]}}
            ]},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    planner = GeminiPlanner(
        api_key="SECRET", model="gemini-2.0-flash", http_client=http_client
    )
    steps = await planner.make_plan("goal")
    assert steps == ["step one", "step two"]
    assert "gemini-2.0-flash:generateContent" in captured["url"]
    assert captured["key"] == "SECRET"


async def test_gemini_planner_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    planner = GeminiPlanner(api_key="bad", http_client=http_client)
    with pytest.raises(httpx.HTTPStatusError):
        await planner.make_plan("goal")
