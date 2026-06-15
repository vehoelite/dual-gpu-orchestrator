"""Owns an autonomous run: planner -> dominant drives -> workers execute.

The dominant is the Phase 1 Agent loop wired to a CoordinationRegistry; the
worker is the Phase 1 Agent + ToolRegistry, built fresh per delegation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from orchestrator.agent import Agent
from orchestrator.coordination import CoordinationRegistry
from orchestrator.plan import Plan

DOMINANT_PROMPT = (
    "You are the DOMINANT orchestrator. You do NOT do work yourself and you "
    "cannot touch files. You drive a checklist to completion by delegating each "
    "step to a worker, reviewing its result, and marking the step done.\n"
    "Use exactly one action per reply. Available actions:\n"
    "::action delegate\nstep: <index>\n---\n<full subtask and context for the worker>\n::end\n"
    "::action mark_done\nstep: <index>\n::end\n"
    "::action revise_plan\n---\n1. ...\n2. ...\n::end\n"
    "When every step is done, emit:\n::action task_complete\n::end\n"
    "Think briefly, then emit one action."
)

WORKER_PROMPT = (
    "You are an autonomous worker. Act using exactly one action block per reply:\n"
    "::action <verb>\nkey: value\n---\noptional body\n::end\n"
    "Verbs: read_file(path), write_file(path + body), list_dir(path), "
    "run_command(cmd). When the subtask is fully done, emit:\n::action done\n::end\n"
    "Think briefly, then emit one action."
)


@dataclass
class RunResult:
    plan: Plan
    dominant_transcript: list[dict]
    worker_results: list[dict]
    stopped_reason: str


class Orchestrator:
    def __init__(
        self,
        planner,
        worker_factory: Callable[[], Agent],
        dominant_client,
        dominant_model: str,
        max_dominant_turns: int = 40,
        no_progress_limit: int = 5,
    ) -> None:
        self.planner = planner
        self.worker_factory = worker_factory
        self.dominant_client = dominant_client
        self.dominant_model = dominant_model
        self.max_dominant_turns = max_dominant_turns
        self.no_progress_limit = no_progress_limit

    async def run(self, goal: str) -> RunResult:
        steps = await self._plan(goal)
        if steps is None:
            return RunResult(Plan(), [], [], "planner_failed")
        plan = Plan.from_descriptions(steps)
        coord = CoordinationRegistry(
            plan, self.worker_factory, no_progress_limit=self.no_progress_limit
        )
        dominant = Agent(
            client=self.dominant_client,
            registry=coord,
            model=self.dominant_model,
            system_prompt=DOMINANT_PROMPT,
            max_steps=self.max_dominant_turns,
            terminal_verbs={"task_complete"},
        )
        task = f"Goal: {goal}\n\n{plan.render()}"
        result = await dominant.run(task)
        # Only the turn cap is renamed; "no_progress", "task_complete", and
        # "no_action" (dominant emitted no parseable action) pass through.
        reason = "max_turns" if result.stopped_reason == "max_steps" else result.stopped_reason
        return RunResult(
            plan=plan,
            dominant_transcript=result.transcript,
            worker_results=coord.worker_results,
            stopped_reason=reason,
        )

    async def _plan(self, goal: str) -> list[str] | None:
        for _ in range(2):  # one retry
            steps = await self.planner.make_plan(goal)
            if steps:
                return steps
        return None
