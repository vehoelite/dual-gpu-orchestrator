"""Dominant coordination verbs over a Plan. ``delegate`` runs a fresh worker
Agent per subtask (mirrors subagent-driven development: no shared history).

``execute`` is async so delegate can await the worker. It also enforces the
no-progress backstop: N consecutive actions with no plan-state change return
("stop", "no_progress"), which ends the dominant loop."""
from __future__ import annotations

import re
from typing import Callable

from orchestrator.plan import Plan, PlanError, parse_checklist
from orchestrator.protocol import Action, serialize_result
from orchestrator.events import NullSink, make_event, plan_event, preview

_RETRY_BANNER = (
    "Reviewer rejected your previous attempt. Do NOT repeat it — take a "
    "different approach."
)


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
        # The in-progress step's live worker conversation. Set by delegate /
        # advance (fresh per new step); resumed by retry; cleared when the
        # owning step is marked done.
        self._active_step: int | None = None
        self._active_transcript: list[dict] | None = None

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
            self._clear_active(index)
            return "ok", self.plan.render()
        if action.verb == "delegate":
            return await self._delegate(action)
        if action.verb == "advance":
            return await self._advance(action)
        if action.verb == "retry":
            return await self._retry(action)
        return "error", f"unknown verb: {action.verb}"

    def _parse_step(self, action: Action) -> int:
        return self._parse_index(action.args.get("step"), "step")

    def _parse_index(self, raw, label: str) -> int:
        if raw is None:
            raise PlanError(f"missing '{label}' arg")
        # Tolerate sloppy refs like "8 (final)" or "step 2" — take the first int.
        match = re.search(r"\d+", str(raw))
        if match is None:
            raise PlanError(f"{label} must contain an integer, got {raw!r}")
        return int(match.group())

    def _clear_active(self, index: int) -> None:
        if self._active_step == index:
            self._active_step = None
            self._active_transcript = None

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
        return await self._run_fresh(index, subtask)

    async def _advance(self, action: Action) -> tuple[str, str]:
        # Mark the current step done AND delegate the next, one turn. Validate
        # EVERYTHING before any mutation so a bad block leaves the plan untouched.
        try:
            done_index = self._parse_index(action.args.get("done"), "done")
            next_index = self._parse_index(action.args.get("step"), "step")
        except PlanError as exc:
            return "error", str(exc)
        subtask = action.body.strip()
        if not subtask:
            return "error", "advance needs the next step's instructions in the body"
        try:
            self.plan._check(done_index)
            self.plan._check(next_index)
        except PlanError as exc:
            return "error", str(exc)
        self.plan.mark_done(done_index)
        self._clear_active(done_index)
        self.plan.mark_in_progress(next_index)
        return await self._run_fresh(next_index, subtask)

    async def _retry(self, action: Action) -> tuple[str, str]:
        # Re-run the SAME step's worker with its context intact (it remembers its
        # failed attempt and the original task). Step stays in_progress.
        try:
            index = self._parse_step(action)
        except PlanError as exc:
            return "error", str(exc)
        if self._active_step != index or self._active_transcript is None:
            return "error", f"no active worker for step {index} to retry"
        note = action.body.strip()
        followup = serialize_result("error", f"{_RETRY_BANNER}\n{note}".rstrip())
        self.sink.emit(make_event(
            "worker_started", step=index, subtask=preview(note), retry=True
        ))
        worker = self.worker_factory()
        result = await worker.resume(self._active_transcript, followup)
        self.sink.emit(make_event(
            "worker_finished", step=index,
            stopped_reason=result.stopped_reason, retry=True,
        ))
        return self._record(index, note, result)

    async def _run_fresh(self, index: int, subtask: str) -> tuple[str, str]:
        self.sink.emit(make_event(
            "worker_started", step=index, subtask=preview(subtask)
        ))
        worker = self.worker_factory()
        result = await worker.run(subtask)
        self.sink.emit(make_event(
            "worker_finished", step=index, stopped_reason=result.stopped_reason
        ))
        return self._record(index, subtask, result)

    def _record(self, index: int, subtask: str, result) -> tuple[str, str]:
        # Remember the live transcript so a retry of this step can resume it.
        self._active_step = index
        self._active_transcript = result.transcript
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
