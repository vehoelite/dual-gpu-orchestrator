"""Pluggable planner: goal -> ordered checklist. Frontier (Gemini) or local.

The key insight (see spec): the plan is the highest-leverage artifact, so it can
use a high-quality frontier model while execution stays local."""
from __future__ import annotations

from typing import Protocol

import httpx

from orchestrator.plan import parse_checklist

_PLANNER_SYSTEM = (
    "You are a planning assistant. Break the user's goal into a short, ordered "
    "checklist of concrete, self-contained steps a developer agent can execute "
    "one at a time. Output ONLY the checklist, one step per line, like:\n"
    "1. first step\n2. second step\nNo preamble, no commentary."
)


class Planner(Protocol):
    async def make_plan(self, goal: str) -> list[str]: ...


class LocalPlanner:
    """Plan with a local LM Studio model via the existing LMStudioClient."""

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    async def make_plan(self, goal: str) -> list[str]:
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": goal},
        ]
        text = await self.client.complete(model=self.model, messages=messages)
        return parse_checklist(text)


class GeminiPlanner:
    """Plan with Google's Gemini API. Key is supplied by the caller (env-sourced);
    it is never hard-coded or committed."""

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def make_plan(self, goal: str) -> list[str]:
        url = f"{self._BASE}/{self.model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": _PLANNER_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": goal}]}],
        }
        resp = await self._client.post(url, params={"key": self.api_key}, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_checklist(text)

    async def aclose(self) -> None:
        await self._client.aclose()
