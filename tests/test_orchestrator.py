from orchestrator.agent import AgentResult
from orchestrator.orchestrator import Orchestrator, RunResult


class StubPlanner:
    def __init__(self, steps):
        self.steps = steps
        self.goal = None

    async def make_plan(self, goal):
        self.goal = goal
        return list(self.steps)


class FakeDominantClient:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    async def complete(self, model, messages, temperature=0.7):
        self.calls.append(messages)
        return self._scripted.pop(0)


class FakeWorker:
    def __init__(self):
        self.received = None

    async def run(self, task):
        self.received = task
        return AgentResult(
            transcript=[{"role": "assistant", "content": f"completed: {task}"}],
            stopped_reason="done",
        )


async def test_full_run_completes():
    planner = StubPlanner(["write file", "verify"])
    dom = FakeDominantClient([
        "::action delegate\nstep: 0\n---\nwrite the file\n::end",
        "::action mark_done\nstep: 0\n::end",
        "::action delegate\nstep: 1\n---\nverify it\n::end",
        "::action mark_done\nstep: 1\n::end",
        "::action task_complete\n::end",
    ])
    orch = Orchestrator(
        planner=planner, worker_factory=lambda: FakeWorker(),
        dominant_client=dom, dominant_model="dom",
    )
    result = await orch.run("build it")
    assert isinstance(result, RunResult)
    assert result.stopped_reason == "task_complete"
    assert result.plan.all_done()
    assert len(result.worker_results) == 2
    assert planner.goal == "build it"


async def test_planner_failure():
    class EmptyPlanner:
        async def make_plan(self, goal):
            return []

    orch = Orchestrator(
        planner=EmptyPlanner(), worker_factory=lambda: FakeWorker(),
        dominant_client=FakeDominantClient([]), dominant_model="dom",
    )
    result = await orch.run("x")
    assert result.stopped_reason == "planner_failed"


async def test_max_turns_backstop():
    planner = StubPlanner(["a"])
    dom = FakeDominantClient([
        "::action delegate\nstep: 0\n---\ngo\n::end",
        "::action delegate\nstep: 0\n---\ngo again\n::end",
        "::action delegate\nstep: 0\n---\nmore\n::end",
    ])
    orch = Orchestrator(
        planner=planner, worker_factory=lambda: FakeWorker(),
        dominant_client=dom, dominant_model="dom",
        max_dominant_turns=2, no_progress_limit=10,
    )
    result = await orch.run("x")
    assert result.stopped_reason == "max_turns"
    assert len(dom.calls) == 2
    # partial state survives a backstop exit
    assert len(result.worker_results) == 2
    assert result.plan.steps[0].status == "in_progress"


async def test_no_progress_backstop():
    planner = StubPlanner(["a"])
    dom = FakeDominantClient([
        "::action mark_done\nstep: 0\n::end",  # progress
        "::action mark_done\nstep: 0\n::end",  # no change -> 1
        "::action mark_done\nstep: 0\n::end",  # no change -> 2 == limit -> stop
        "::action task_complete\n::end",       # not reached
    ])
    orch = Orchestrator(
        planner=planner, worker_factory=lambda: FakeWorker(),
        dominant_client=dom, dominant_model="dom",
        max_dominant_turns=10, no_progress_limit=2,
    )
    result = await orch.run("x")
    assert result.stopped_reason == "no_progress"
