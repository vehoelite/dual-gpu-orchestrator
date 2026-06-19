"""Dominant coordination verbs over a Plan. ``delegate`` runs a fresh worker
Agent per subtask (mirrors subagent-driven development: no shared history).

``execute`` is async so delegate can await the worker. It also enforces the
no-progress backstop: N consecutive actions with no plan-state change return
("stop", "no_progress"), which ends the dominant loop."""
from __future__ import annotations

import re
from typing import Callable

from orchestrator.plan import Plan, PlanError, parse_checklist
from orchestrator.protocol import Action
from orchestrator.events import NullSink, make_event, plan_event, preview


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
        sink=None,
    ) -> None:
        self.plan = plan
        self.worker_factory = worker_factory
        self.no_progress_limit = no_progress_limit
        self.no_progress_count = 0
        self.worker_results: list[dict] = []
        self.sink = sink or NullSink()

    async def execute(self, action: Action) -> tuple[str, str]:
        before = self.plan.signature()
        status, message = await self._dispatch(action)
        if self.plan.signature() == before:
            self.no_progress_count += 1
        else:
            self.no_progress_count = 0
            self.sink.emit(plan_event(self.plan))
        if self.no_progress_count >= self.no_progress_limit:
            self.sink.emit(make_event("no_progress"))
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
        # Tolerate sloppy refs like "8 (final)" or "step 2" — take the first int.
        match = re.search(r"\d+", str(raw))
        if match is None:
            raise PlanError(f"step must contain an integer, got {raw!r}")
        return int(match.group())

    async def _delegate(self, action: Action) -> tuple[str, str]:
        # Validate everything BEFORE mutating the plan, so a failed delegate
        # never leaves a step stuck in "in_progress".
        try:
            index = self._parse_step(action)
        except PlanError as exc:
            return "error", str(exc)
        subtask = action.body.strip()
        if not subtask:
            return "error", "delegate needs a subtask in the body"
        try:
            self.plan.mark_in_progress(index)
        except PlanError as exc:
            return "error", str(exc)
        self.sink.emit(make_event(
            "worker_started", step=index, subtask=preview(subtask)
        ))
        worker = self.worker_factory()
        result = await worker.run(subtask)
        self.sink.emit(make_event(
            "worker_finished", step=index, stopped_reason=result.stopped_reason
        ))
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
