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


from orchestrator.agent import Agent
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


async def test_orchestrator_emits_full_event_stream(tmp_path):
    sink = RecordingSink()
    planner = StubPlanner(["create out.txt"])
    dom = FakeDominantClient([
        "::action delegate\nstep: 0\n---\nwrite out.txt with hi\n::end",
        "::action mark_done\nstep: 0\n::end",
        "::action task_complete\n::end",
    ])

    def worker_factory():
        return Agent(
            client=FakeDominantClient([
                "::action write_file\npath: out.txt\n---\nhi\n::end",
                "::action done\n::end",
            ]),
            registry=ToolRegistry(Sandbox(tmp_path), command_timeout=10.0),
            model="w", system_prompt="s", max_steps=5,
            sink=sink, agent_label="worker",
        )

    orch = Orchestrator(
        planner=planner, worker_factory=worker_factory,
        dominant_client=dom, dominant_model="dom", sink=sink,
    )
    result = await orch.run("build it")
    assert result.stopped_reason == "task_complete"

    types = [e["type"] for e in sink.events]
    assert types[0] == "plan"  # seed plan emitted first
    assert "worker_started" in types and "worker_finished" in types
    assert types.index("worker_started") < types.index("worker_finished")
    assert any(
        e["type"] == "message" and e["agent"] == "worker" for e in sink.events
    )
    assert any(
        e["type"] == "action" and e["agent"] == "dominant" and e["verb"] == "delegate"
        for e in sink.events
    )
    assert any(e["type"] == "action" and e["verb"] == "task_complete" for e in sink.events)
    assert (tmp_path / "out.txt").read_text() == "hi"
