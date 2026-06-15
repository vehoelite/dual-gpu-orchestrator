"""Headless autonomous run.

Usage:
    python -m orchestrator.cli "<goal>" <project_folder>

Picks dominant/worker from LM Studio's loaded models (first two). Planner is
chosen by Config.planner; the Gemini key is read from GEMINI_API_KEY (never
committed). Runs to completion with no human input.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from orchestrator.agent import Agent
from orchestrator.config import Config
from orchestrator.llm_client import LMStudioClient
from orchestrator.orchestrator import Orchestrator, WORKER_PROMPT
from orchestrator.planner import GeminiPlanner, LocalPlanner
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


def _build_planner(cfg: Config, client: LMStudioClient):
    if cfg.planner == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            return GeminiPlanner(api_key=key, model=cfg.gemini_model)
        if not cfg.planner_fallback_local:
            raise SystemExit("GEMINI_API_KEY not set and fallback disabled")
        print("GEMINI_API_KEY not set; falling back to local planner")
    return LocalPlanner(client=client, model=cfg.dominant_model)


async def main() -> int:
    if len(sys.argv) < 3:
        print('usage: python -m orchestrator.cli "<goal>" <project_folder>')
        return 2
    goal = sys.argv[1]
    project = Path(sys.argv[2])
    project.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)
    try:
        models = await client.list_models()
        if not models:
            raise SystemExit("no models loaded in LM Studio")
        cfg.dominant_model = cfg.dominant_model or models[0]
        cfg.worker_model = cfg.worker_model or (models[1] if len(models) > 1 else models[0])
        print(f"dominant={cfg.dominant_model} worker={cfg.worker_model}")

        def worker_factory() -> Agent:
            return Agent(
                client=client,
                registry=ToolRegistry(Sandbox(project), cfg.command_timeout),
                model=cfg.worker_model,
                system_prompt=WORKER_PROMPT,
                max_steps=cfg.max_steps,
            )

        planner = _build_planner(cfg, client)
        orch = Orchestrator(
            planner=planner,
            worker_factory=worker_factory,
            dominant_client=client,
            dominant_model=cfg.dominant_model,
            max_dominant_turns=cfg.max_dominant_turns,
            no_progress_limit=cfg.no_progress_limit,
        )
        result = await orch.run(goal)
        print(f"\nStopped: {result.stopped_reason}")
        print(result.plan.render())
    finally:
        await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
