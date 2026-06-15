"""Manual end-to-end smoke check against a running LM Studio.

Usage:
    python -m orchestrator.smoke <project_folder> [model_id]

Picks the first available model if none is given, then asks the agent to create
hello.txt containing "hello world" in the project folder.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from orchestrator.agent import Agent
from orchestrator.config import Config
from orchestrator.llm_client import LMStudioClient
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry

SYSTEM_PROMPT = (
    "You are an autonomous worker. Act using exactly one action block per reply:\n"
    "::action <verb>\nkey: value\n---\noptional body\n::end\n"
    "Verbs: read_file(path), write_file(path + body), list_dir(path), "
    "run_command(cmd). When the task is fully done, emit:\n::action done\n::end\n"
    "Think briefly in prose, then emit one action."
)


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m orchestrator.smoke <project_folder> [model_id]")
        return 2

    project = Path(sys.argv[1])
    project.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)

    try:
        model = sys.argv[2] if len(sys.argv) > 2 else (await client.list_models())[0]
        print(f"Using model: {model}")
        agent = Agent(
            client=client,
            registry=ToolRegistry(Sandbox(project), cfg.command_timeout),
            model=model,
            system_prompt=SYSTEM_PROMPT,
            max_steps=cfg.max_steps,
        )
        result = await agent.run(
            'Create a file named hello.txt containing exactly "hello world".'
        )
        print(f"Stopped: {result.stopped_reason}")
        target = project / "hello.txt"
        print(f"hello.txt exists: {target.exists()}")
        if target.exists():
            print(f"contents: {target.read_text()!r}")
    finally:
        await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
