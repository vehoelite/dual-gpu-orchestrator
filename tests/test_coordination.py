from orchestrator.agent import AgentResult
from orchestrator.coordination import CoordinationRegistry
from orchestrator.plan import Plan
from orchestrator.protocol import Action


class FakeWorker:
    def __init__(self, report="done it", stopped_reason="done"):
        self.report = report
        self.stopped_reason = stopped_reason
        self.received = None
        self.resumed = None  # (prior_transcript, followup) when resume() is used

    async def run(self, task):
        self.received = task
        return AgentResult(
            transcript=[
                {"role": "user", "content": task},
                {"role": "assistant", "content": self.report},
            ],
            stopped_reason=self.stopped_reason,
        )

    async def resume(self, prior_transcript, followup):
        self.resumed = (prior_transcript, followup)
        return AgentResult(
            transcript=list(prior_transcript)
            + [
                {"role": "user", "content": followup},
                {"role": "assistant", "content": self.report},
            ],
            stopped_reason=self.stopped_reason,
        )


async def test_set_plan_initializes():
    plan = Plan()
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("set_plan", {}, "1. a\n2. b"))
    assert status == "ok"
    assert [s.description for s in plan.steps] == ["a", "b"]


async def test_delegate_runs_fresh_worker_and_marks_in_progress():
    plan = Plan.from_descriptions(["do the thing", "later"])
    worker = FakeWorker(report="all done", stopped_reason="done")
    coord = CoordinationRegistry(plan, worker_factory=lambda: worker)
    status, msg = await coord.execute(
        Action("delegate", {"step": "0"}, "do the thing now")
    )
    assert status == "ok"
    assert worker.received == "do the thing now"
    assert plan.steps[0].status == "in_progress"
    assert "all done" in msg
    assert coord.worker_results[0]["stopped_reason"] == "done"


async def test_delegate_empty_body_is_error_and_plan_unchanged():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("delegate", {"step": "0"}, "  "))
    assert status == "error"
    assert plan.steps[0].status == "pending"  # not left stuck in_progress


async def test_delegate_bad_index_is_error_and_plan_unchanged():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("delegate", {"step": "9"}, "do it"))
    assert status == "error"
    assert plan.steps[0].status == "pending"


async def test_mark_done_updates_plan():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("mark_done", {"step": "0"}, ""))
    assert status == "ok"
    assert plan.steps[0].status == "done"


async def test_bad_step_index_is_error():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("mark_done", {"step": "9"}, ""))
    assert status == "error"


async def test_unknown_verb_is_error():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("frobnicate", {}, ""))
    assert status == "error"
    assert "frobnicate" in msg


async def test_revise_plan_replaces():
    plan = Plan.from_descriptions(["a", "b"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("revise_plan", {}, "1. x\n2. y\n3. z"))
    assert status == "ok"
    assert [s.description for s in plan.steps] == ["x", "y", "z"]


async def test_no_progress_returns_stop():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None, no_progress_limit=3)
    for _ in range(2):
        status, msg = await coord.execute(Action("frobnicate", {}, ""))
        assert status == "error"
    status, msg = await coord.execute(Action("frobnicate", {}, ""))
    assert status == "stop"
    assert msg == "no_progress"


async def test_progress_resets_counter():
    plan = Plan.from_descriptions(["a", "b"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None, no_progress_limit=2)
    await coord.execute(Action("frobnicate", {}, ""))  # no change -> count 1
    await coord.execute(Action("mark_done", {"step": "0"}, ""))  # change -> reset
    assert coord.no_progress_count == 0


async def test_step_tolerates_trailing_text(tmp_path):
    plan = Plan.from_descriptions(["a", "b"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("mark_done", {"step": "1 (final)"}, ""))
    assert status == "ok"
    assert plan.steps[1].status == "done"


# --- advance: mark current done + delegate next, one turn -------------------

async def test_advance_marks_done_and_delegates_next():
    plan = Plan.from_descriptions(["first", "second"])
    plan.mark_in_progress(0)
    worker = FakeWorker(report="second done")
    coord = CoordinationRegistry(plan, worker_factory=lambda: worker)
    status, msg = await coord.execute(
        Action("advance", {"done": "0", "step": "1"}, "do second")
    )
    assert status == "ok"
    assert plan.steps[0].status == "done"
    assert plan.steps[1].status == "in_progress"
    assert worker.received == "do second"
    assert "second done" in msg


async def test_advance_empty_body_is_error_and_plan_unchanged():
    plan = Plan.from_descriptions(["a", "b"])
    plan.mark_in_progress(0)
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(
        Action("advance", {"done": "0", "step": "1"}, "   ")
    )
    assert status == "error"
    assert plan.steps[0].status == "in_progress"  # not marked done
    assert plan.steps[1].status == "pending"


async def test_advance_bad_next_index_is_error_and_no_mutation():
    plan = Plan.from_descriptions(["a", "b"])
    plan.mark_in_progress(0)
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(
        Action("advance", {"done": "0", "step": "9"}, "go")
    )
    assert status == "error"
    assert plan.steps[0].status == "in_progress"  # done not applied


async def test_advance_bad_done_index_is_error_and_no_mutation():
    plan = Plan.from_descriptions(["a", "b"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(
        Action("advance", {"done": "9", "step": "1"}, "go")
    )
    assert status == "error"
    assert plan.steps[1].status == "pending"  # next not started


async def test_advance_resets_no_progress():
    plan = Plan.from_descriptions(["a", "b"])
    plan.mark_in_progress(0)
    coord = CoordinationRegistry(
        plan, worker_factory=lambda: FakeWorker(), no_progress_limit=5
    )
    coord.no_progress_count = 3
    await coord.execute(Action("advance", {"done": "0", "step": "1"}, "go"))
    assert coord.no_progress_count == 0


# --- retry: re-run the SAME step's worker with its context preserved --------

async def test_retry_resumes_active_worker_with_rejection_note():
    plan = Plan.from_descriptions(["a"])
    first = FakeWorker(report="attempt one")
    coord = CoordinationRegistry(plan, worker_factory=lambda: first)
    await coord.execute(Action("delegate", {"step": "0"}, "do a"))

    second = FakeWorker(report="attempt two")
    coord.worker_factory = lambda: second
    status, msg = await coord.execute(
        Action("retry", {"step": "0"}, "price field missing; fix it")
    )
    assert status == "ok"
    assert second.resumed is not None
    prior, followup = second.resumed
    assert prior[-1]["content"] == "attempt one"  # resumed from the failed attempt
    assert "price field missing; fix it" in followup
    assert "attempt two" in msg
    assert plan.steps[0].status == "in_progress"  # retry does not mark done


async def test_retry_without_active_worker_is_error():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: None)
    status, msg = await coord.execute(Action("retry", {"step": "0"}, "fix it"))
    assert status == "error"


async def test_retry_for_non_active_step_is_error():
    plan = Plan.from_descriptions(["a", "b"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: FakeWorker())
    await coord.execute(Action("delegate", {"step": "0"}, "do a"))
    status, msg = await coord.execute(Action("retry", {"step": "1"}, "fix it"))
    assert status == "error"


async def test_retry_increments_no_progress():
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(
        plan, worker_factory=lambda: FakeWorker(), no_progress_limit=5
    )
    await coord.execute(Action("delegate", {"step": "0"}, "do a"))  # sig change
    assert coord.no_progress_count == 0
    await coord.execute(Action("retry", {"step": "0"}, "again"))  # no sig change
    assert coord.no_progress_count == 1


async def test_advance_clears_active_worker_so_old_step_cannot_retry():
    plan = Plan.from_descriptions(["a", "b"])
    coord = CoordinationRegistry(plan, worker_factory=lambda: FakeWorker())
    await coord.execute(Action("delegate", {"step": "0"}, "do a"))
    await coord.execute(Action("advance", {"done": "0", "step": "1"}, "do b"))
    # Active worker is now step 1; retrying the finished step 0 must error.
    status, msg = await coord.execute(Action("retry", {"step": "0"}, "redo a"))
    assert status == "error"


async def test_retry_emits_worker_events_with_retry_flag():
    sink = _RecordingSink()
    plan = Plan.from_descriptions(["a"])
    coord = CoordinationRegistry(
        plan, worker_factory=lambda: FakeWorker(), sink=sink
    )
    await coord.execute(Action("delegate", {"step": "0"}, "do a"))
    sink.events.clear()
    await coord.execute(Action("retry", {"step": "0"}, "again"))
    started = next(e for e in sink.events if e["type"] == "worker_started")
    assert started.get("retry") is True


from orchestrator.agent import AgentResult
from orchestrator.coordination import CoordinationRegistry
from orchestrator.plan import Plan
from orchestrator.protocol import Action


class _RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


class _FakeWorker:
    async def run(self, task):
        return AgentResult(
            transcript=[{"role": "assistant", "content": "did it"}],
            stopped_reason="done",
        )


async def test_coordination_emits_worker_and_plan_events():
    sink = _RecordingSink()
    plan = Plan.from_descriptions(["do a thing"])
    coord = CoordinationRegistry(
        plan, worker_factory=lambda: _FakeWorker(), sink=sink
    )
    await coord.execute(Action("delegate", {"step": "0"}, "go do it"))
    types = [e["type"] for e in sink.events]
    assert "worker_started" in types
    assert "worker_finished" in types
    assert "plan" in types
    assert types.index("worker_started") < types.index("worker_finished")
    started = next(e for e in sink.events if e["type"] == "worker_started")
    assert started["step"] == 0
