"""Dominant coordination verbs over a Plan. ``delegate`` runs a fresh worker
Agent per subtask (mirrors subagent-driven development: no shared history).

``execute`` is async so delegate can await the worker. It also enforces the
no-progress backstop: N consecutive actions with no plan-state change return
("stop", "no_progress"), which ends the dominant loop."""
from __future__ import annotations

from typing import Callable

from orchestrator.plan import Plan, PlanError, parse_checklist
from orchestrator.protocol import Action


def _last_assistant(transcript: list[dict]) -> str:
    for msg in reversed(transcript):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


class CoordinationRegistry:
    def __init__(
        self,
        plan: Plan,
        worker_factory: Callable[[], object],
        no_progress_limit: int = 5,
    ) -> None:
        self.plan = plan
        self.worker_factory = worker_factory
        self.no_progress_limit = no_progress_limit
        self.no_progress_count = 0
        self.worker_results: list[dict] = []

    async def execute(self, action: Action) -> tuple[str, str]:
        before = self.plan.signature()
        status, message = await self._dispatch(action)
        if self.plan.signature() == before:
            self.no_progress_count += 1
        else:
            self.no_progress_count = 0
        if self.no_progress_count >= self.no_progress_limit:
            return "stop", "no_progress"
        return status, message

    async def _dispatch(self, action: Action) -> tuple[str, str]:
        if action.verb in ("set_plan", "revise_plan"):
            steps = parse_checklist(action.body)
            if not steps:
                return "error", f"{action.verb} needs a checklist in the body"
            self.plan.revise(steps)
            return "ok", self.plan.render()
        if action.verb == "mark_done":
            try:
                index = self._parse_step(action)
                self.plan.mark_done(index)
            except PlanError as exc:
                return "error", str(exc)
            return "ok", self.plan.render()
        if action.verb == "delegate":
            return await self._delegate(action)
        return "error", f"unknown verb: {action.verb}"

    def _parse_step(self, action: Action) -> int:
        raw = action.args.get("step")
        if raw is None:
            raise PlanError("missing 'step' arg")
        try:
            return int(raw)
        except ValueError:
            raise PlanError(f"step must be an integer, got {raw!r}")

    async def _delegate(self, action: Action) -> tuple[str, str]:
        try:
            index = self._parse_step(action)
            self.plan.mark_in_progress(index)
        except PlanError as exc:
            return "error", str(exc)
        subtask = action.body.strip()
        if not subtask:
            return "error", "delegate needs a subtask in the body"
        worker = self.worker_factory()
        result = await worker.run(subtask)
        report = _last_assistant(result.transcript)
        self.worker_results.append(
            {
                "step": index,
                "subtask": subtask,
                "stopped_reason": result.stopped_reason,
                "report": report,
            }
        )
        message = (
            f"worker finished step {index} (stopped: {result.stopped_reason}).\n"
            f"worker report:\n{report}\n\n{self.plan.render()}"
        )
        return "ok", message
