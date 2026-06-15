"""Headless autonomous run.

Usage:
    python -m orchestrator.cli "<goal>" <project_folder>
        [--dominant <id-or-substring>] [--worker <id-or-substring>]

Without --dominant/--worker, roles are assigned by LM Studio's load order
(first = dominant, second = worker), which is arbitrary. Pin them explicitly so
the stronger model orchestrates, e.g. --dominant 9b --worker 4b. Planner is
chosen by Config.planner; the Gemini key is read from GEMINI_API_KEY (never
committed). Runs to completion with no human input.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from orchestrator.agent import Agent
from orchestrator.config import Config
from orchestrator.llm_client import LMStudioClient
from orchestrator.orchestrator import Orchestrator, WORKER_PROMPT
from orchestrator.planner import GeminiPlanner, LocalPlanner
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


def _select_models(
    models: list[str], dominant: str | None = None, worker: str | None = None
) -> tuple[str, str]:
    """Resolve dominant/worker model ids from the loaded models.

    A hint matches the first model whose id contains it (case-insensitive). With
    no hint, the dominant defaults to the first model and the worker to the
    second (or the first if only one is loaded). Raises ValueError if a hint
    matches nothing."""

    def pick(hint: str | None, default: str) -> str:
        if not hint:
            return default
        for model in models:
            if hint.lower() in model.lower():
                return model
        raise ValueError(
            f"no loaded model matches {hint!r}; available: {', '.join(models)}"
        )

    second = models[1] if len(models) > 1 else models[0]
    return pick(dominant, models[0]), pick(worker, second)


def _build_planner(cfg: Config, client: LMStudioClient):
    if cfg.planner == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            return GeminiPlanner(api_key=key, model=cfg.gemini_model)
        if not cfg.planner_fallback_local:
            raise SystemExit("GEMINI_API_KEY not set and fallback disabled")
        print("GEMINI_API_KEY not set; falling back to local planner")
    return LocalPlanner(client=client, model=cfg.dominant_model)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m orchestrator.cli")
    parser.add_argument("goal", help="the goal to accomplish")
    parser.add_argument("project", help="project folder (the sandbox)")
    parser.add_argument(
        "--dominant", default=None,
        help="model id or substring for the dominant/orchestrator (default: first loaded)",
    )
    parser.add_argument(
        "--worker", default=None,
        help="model id or substring for the worker (default: second loaded)",
    )
    return parser.parse_args()


async def main() -> int:
    args = _parse_args()
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)
    try:
        models = await client.list_models()
        if not models:
            raise SystemExit("no models loaded in LM Studio")
        try:
            cfg.dominant_model, cfg.worker_model = _select_models(
                models, args.dominant, args.worker
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
        print(f"dominant={cfg.dominant_model}\nworker={cfg.worker_model}")
        if not args.dominant and not args.worker and len(models) > 1:
            print("(roles assigned by load order; override with --dominant/--worker)")

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
        result = await orch.run(args.goal)
        print(f"\nStopped: {result.stopped_reason}")
        print(result.plan.render())
    finally:
        await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
