"""Async client for LM Studio's OpenAI-compatible API. Plain completions only —
we never send the ``tools`` parameter (see spec section 5)."""
from __future__ import annotations

import httpx


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def list_models(self) -> list[str]:
        resp = await self._client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        resp = await self._client.post(
            f"{self.base_url}/chat/completions", json=payload
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
