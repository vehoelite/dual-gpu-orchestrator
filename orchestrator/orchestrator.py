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
    "cannot touch files. Drive the checklist to completion by delegating each "
    "step to a worker, reviewing its report, and marking the step done.\n"
    "\n"
    "Each reply: think in ONE short sentence, then emit EXACTLY ONE action block "
    "and nothing after it. An action block spans MULTIPLE LINES with real "
    "newlines: the verb on the ::action line, each arg on its own 'name: value' "
    "line (no quotes, no '='), an optional body after a line that is only ---, "
    "closed by a line that is only ::end. Never put the whole block on one line.\n"
    "\n"
    "CRITICAL: each worker is a FRESH agent with NO memory of other workers or "
    "earlier steps. When a worker's report contains something a later step needs "
    "(a headline, URL, value, or file contents), you MUST copy that EXACT text "
    "into the later delegate's body. Never write 'the headline' or 'the result' - "
    "paste the literal content, because the next worker cannot see it otherwise.\n"
    "\n"
    "Efficiency: to write a file, tell the worker to use write_file with the "
    "literal content in the body (never echo). Do not create separate steps just "
    "to verify or re-read a file you already had written.\n"
    "\n"
    "Delegate a step (put full, self-contained instructions in the body):\n"
    "::action delegate\n"
    "step: 0\n"
    "---\n"
    "Create a file hello.py containing exactly: print(\"hello world\")\n"
    "::end\n"
    "\n"
    "Mark a step done after reviewing the worker's report:\n"
    "::action mark_done\n"
    "step: 0\n"
    "::end\n"
    "\n"
    "Replace the remaining steps if the plan needs to change:\n"
    "::action revise_plan\n"
    "---\n"
    "1. first step\n"
    "2. second step\n"
    "::end\n"
    "\n"
    "When EVERY step is done, finish the run:\n"
    "::action task_complete\n"
    "::end"
)

WORKER_PROMPT = (
    "You are an autonomous worker. Each reply: think in ONE short sentence, then "
    "emit EXACTLY ONE action block and nothing after it.\n"
    "If you only DESCRIBE an action without emitting the ::action block, nothing "
    "happens — you MUST always include the block.\n"
    "\n"
    "An action block spans MULTIPLE LINES with real newlines: the verb on the "
    "::action line, each arg on its own 'name: value' line (no quotes, no '='), "
    "the body after a line that is only ---, closed by a line that is only ::end. "
    "Never put the whole block on one line.\n"
    "\n"
    "Verbs and their args:\n"
    "  read_file    (arg: path)\n"
    "  write_file   (arg: path; the file contents go in the body)\n"
    "  list_dir     (arg: path)\n"
    "  run_command  (arg: cmd)\n"
    "\n"
    "Use write_file to create/overwrite a file and read_file to read one. Do NOT "
    "use run_command with echo/cat/type for file I/O - that wastes turns. Reserve "
    "run_command for things the other verbs cannot do. Do NOT run extra commands "
    "to 'verify' your work unless the task explicitly asks for verification.\n"
    "\n"
    "Example - write a file:\n"
    "::action write_file\n"
    "path: hello.py\n"
    "---\n"
    "print(\"hello world\")\n"
    "::end\n"
    "\n"
    "When the subtask is fully done, emit:\n"
    "::action done\n"
    "::end"
)

RESEARCH_HINT = (
    "\n\nYou also have web/tool research via MCP. Use it for facts you don't know:\n"
    "::action research\n"
    "query: latest NVIDIA data-center news with sources\n"
    "::end\n"
    "The result is a synthesized answer from external tools."
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
