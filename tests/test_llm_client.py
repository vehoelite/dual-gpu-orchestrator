import httpx
import pytest

from orchestrator.llm_client import LMStudioClient


def _client_with(handler) -> LMStudioClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return LMStudioClient(base_url="http://test/v1", timeout=5.0, http_client=http_client)


async def test_list_models_parses_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})

    client = _client_with(handler)
    assert await client.list_models() == ["model-a", "model-b"]


async def test_token_sets_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = LMStudioClient(base_url="http://test/v1", http_client=http_client, token="sk-lm-test")
    await client.list_models()
    assert seen["auth"] == "Bearer sk-lm-test"


async def test_no_token_no_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": []})

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = LMStudioClient(base_url="http://test/v1", http_client=http_client)
    await client.list_models()
    assert seen["auth"] is None


async def test_complete_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = request.read().decode()
        assert "tools" not in body  # we never send the tools param
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello from model"}}]},
        )

    client = _client_with(handler)
    out = await client.complete(
        model="model-a",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out == "hello from model"


async def test_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.complete(model="m", messages=[{"role": "user", "content": "x"}])
