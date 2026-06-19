# Phase 4: Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live web dashboard over the existing autonomous engine — configure a run, watch the conversation/actions/plan update live over a WebSocket, and kill the run instantly.

**Architecture:** Thread an *optional* event sink through the existing `agent`/`coordination`/`orchestrator` loops (default a no-op `NullSink`, so the headless `cli.py` and all current tests are unchanged). The sink emits **turn-level** events. A `RunManager` owns the single active run (its asyncio task + an `EventBus`); a FastAPI app exposes REST + a `/ws` WebSocket that replays the current run's buffered events then streams live ones. A vanilla, build-free SPA renders it. The kill switch is a hard `task.cancel()`.

**Tech Stack:** Python 3.11+, `httpx` (existing), **`fastapi` + `uvicorn[standard]` (new)**, `pytest` + `pytest-asyncio`. Vanilla HTML/CSS/JS frontend — no build step, no CDN.

**Reference spec:** `docs/superpowers/specs/2026-06-19-phase4-web-ui-design.md`

## Global Constraints

- Python 3.11+. `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed).
- **Do NOT modify `llm_client.py`** — it stays `stream: False`. Events are turn-level only.
- **Every new sink parameter defaults to `NullSink()`** — headless behavior and the existing no-sink test suite must stay green (backward-compat gate).
- One active run at a time. A second `POST /api/run` while running → HTTP 409.
- Kill switch = hard `asyncio` task cancellation; clean up clients in `finally`; emit exactly one terminal event (`run_aborted`).
- Server binds to `127.0.0.1` (from `Config.host`/`Config.port`).
- Frontend: plain HTML/CSS/JS only. No build tooling, no CDN (offline-capable).
- All tests run on Windows via `.venv\Scripts\python.exe -m pytest`. Tool cwd may be `C:\Users\jacob`; use absolute paths or run from the repo root `C:\Users\jacob\dual-gpu-orchestrator`. Local git identity if needed: `git config user.email "vehoelite@gmail.com"; git config user.name "Jacob"`.
- Event payloads are plain JSON-serializable dicts with a `type` and `ts`. Long text fields are truncated via `preview()` before emission.

---

## File Structure

```
orchestrator/
  events.py        # NEW: make_event(), preview(), plan_event(), NullSink, EventBus
  run_manager.py   # NEW: RunManager (single-run asyncio task lifecycle + EventBus)
  server.py        # NEW: FastAPI app, RunManager wiring, REST + /ws, entry point
  agent.py         # MODIFY: optional sink + agent_label; emit message/action/result/parse_error
  coordination.py  # MODIFY: optional sink; emit worker_started/worker_finished/plan/no_progress
  orchestrator.py  # MODIFY: optional sink; emit seed plan; wire sink into dominant + coordination
  config.py        # MODIFY: host, port
  static/          # NEW: index.html, app.js, style.css (vanilla SPA)
pyproject.toml     # MODIFY: add fastapi + uvicorn[standard]
tests/
  test_events.py          # NEW
  test_run_manager.py     # NEW
  test_server.py          # NEW
  test_agent.py           # MODIFY (append event tests)
  test_coordination.py    # MODIFY (append event test)
  test_orchestrator.py    # MODIFY (append integration event test)
  test_config.py          # MODIFY (append host/port test)
```

**Shared interfaces (define once, reuse — do not rename):**
- `make_event(event_type: str, **fields) -> dict` — returns `{"type": event_type, "ts": <float>, **fields}`.
- `preview(text: str, cap: int = 4000) -> str` — truncates long text.
- `plan_event(plan) -> dict` — `make_event("plan", steps=[{"index","description","status"}...], done=int, total=int)`; duck-typed on `plan.steps`.
- `NullSink` with `emit(event: dict) -> None` (no-op). `EventBus` with `emit(event)`, `subscribe() -> asyncio.Queue`, `unsubscribe(queue)`, `replay() -> list[dict]`, `reset()`. **`emit` is synchronous** (non-blocking `put_nowait`).
- `Agent.__init__(..., sink=None, agent_label="agent")` — `sink or NullSink()`.
- `CoordinationRegistry.__init__(plan, worker_factory, no_progress_limit=5, sink=None)`.
- `Orchestrator.__init__(..., sink=None)`.
- `RunManager` with `bus: EventBus`, `is_running -> bool`, `start(factory)`, `async stop()`. `factory` is `async def (bus: EventBus) -> None`.
- `Config` gains `host: str = "127.0.0.1"`, `port: int = 8000`.
- `server.build_run_factory(params: dict, cfg: Config) -> Callable[[EventBus], Awaitable[None]]` — the production run; a monkeypatch seam for tests.

---

## Task 1: Event layer (`events.py`)

**Files:**
- Create: `orchestrator/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces: `make_event`, `preview`, `plan_event`, `NullSink`, `EventBus` (signatures above).

- [ ] **Step 1: Write the failing tests**

`tests/test_events.py`:
```python
import asyncio

from orchestrator.events import EventBus, NullSink, make_event, plan_event, preview


def test_make_event_has_type_and_ts():
    ev = make_event("message", agent="worker", text="hi")
    assert ev["type"] == "message"
    assert isinstance(ev["ts"], float)
    assert ev["agent"] == "worker"
    assert ev["text"] == "hi"


def test_preview_truncates():
    assert preview("abcdef", cap=3) == "abc"
    assert preview("ab", cap=3) == "ab"


def test_plan_event_shape():
    class FakeStep:
        def __init__(self, d, s):
            self.description = d
            self.status = s

    class FakePlan:
        steps = [FakeStep("a", "done"), FakeStep("b", "pending")]

    ev = plan_event(FakePlan())
    assert ev["type"] == "plan"
    assert ev["total"] == 2
    assert ev["done"] == 1
    assert ev["steps"] == [
        {"index": 0, "description": "a", "status": "done"},
        {"index": 1, "description": "b", "status": "pending"},
    ]


async def test_nullsink_emit_is_noop():
    assert NullSink().emit(make_event("x")) is None


async def test_bus_fans_out_and_buffers():
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.emit(make_event("a"))
    assert (await q1.get())["type"] == "a"
    assert (await q2.get())["type"] == "a"
    assert [e["type"] for e in bus.replay()] == ["a"]


async def test_bus_unsubscribe_stops_delivery():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    bus.emit(make_event("a"))
    assert q.empty()
    # but the buffer still records it
    assert len(bus.replay()) == 1


async def test_bus_reset_clears_buffer_keeps_subscribers():
    bus = EventBus()
    q = bus.subscribe()
    bus.emit(make_event("old"))
    bus.reset()
    assert bus.replay() == []
    bus.emit(make_event("new"))
    assert (await q.get())["type"] == "new"


async def test_bus_drops_for_full_slow_consumer():
    bus = EventBus(queue_size=1)
    q = bus.subscribe()
    bus.emit(make_event("a"))
    bus.emit(make_event("b"))  # q full -> dropped, must not raise
    assert q.qsize() == 1
    assert len(bus.replay()) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.events'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/events.py`:
```python
"""Live-observation layer: turn-level events plus an async fan-out bus.

Events are plain JSON-serializable dicts. NullSink is the default no-op sink so
instrumentation is invisible when no UI is attached (keeps the headless engine
and the existing tests unchanged)."""
from __future__ import annotations

import asyncio
import time
from collections import deque


def make_event(event_type: str, **fields) -> dict:
    """Build an event dict with a type and a timestamp."""
    return {"type": event_type, "ts": time.time(), **fields}


def preview(text: str, cap: int = 4000) -> str:
    """Truncate long text so event frames stay small."""
    text = "" if text is None else str(text)
    return text[:cap]


def plan_event(plan) -> dict:
    """Snapshot a Plan (duck-typed on .steps) as a 'plan' event."""
    steps = [
        {"index": i, "description": s.description, "status": s.status}
        for i, s in enumerate(plan.steps)
    ]
    done = sum(1 for s in plan.steps if s.status == "done")
    return make_event("plan", steps=steps, done=done, total=len(plan.steps))


class NullSink:
    """A sink that discards events. Default everywhere instrumentation exists."""

    def emit(self, event: dict) -> None:
        return None


class EventBus:
    """Synchronous, non-blocking fan-out to per-subscriber queues, plus a bounded
    ring buffer of the current run's events for replay on (re)connect."""

    def __init__(self, queue_size: int = 1000, buffer_size: int = 2000) -> None:
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue] = set()
        self._buffer: deque[dict] = deque(maxlen=buffer_size)

    def emit(self, event: dict) -> None:
        self._buffer.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow consumer: drop rather than block the run

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def replay(self) -> list[dict]:
        return list(self._buffer)

    def reset(self) -> None:
        # A new run starts everyone from a clean slate: clear the replay buffer
        # AND drain any stale prior-run events still sitting in subscriber queues.
        self._buffer.clear()
        for q in self._subscribers:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_events.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/events.py tests/test_events.py
git commit -m "feat: add event layer (make_event, preview, plan_event, NullSink, EventBus)"
```

---

## Task 2: Instrument the agent loop (`agent.py`)

**Files:**
- Modify: `orchestrator/agent.py`
- Test: `tests/test_agent.py` (append)

**Interfaces:**
- Consumes: `make_event`, `preview`, `NullSink` from `orchestrator.events`.
- Produces: `Agent(..., sink=None, agent_label="agent")` emitting `message`, `action`, `result`, `parse_error` events.

- [ ] **Step 1: Append the failing tests to `tests/test_agent.py`**

```python
class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


async def test_agent_emits_event_sequence(tmp_path):
    sink = RecordingSink()
    registry = ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)
    agent = Agent(
        client=FakeClient([
            "::action write_file\npath: a.txt\n---\nhi\n::end",
            "::action done\n::end",
        ]),
        registry=registry, model="m", system_prompt="s", max_steps=5,
        sink=sink, agent_label="worker",
    )
    await agent.run("t")
    assert [e["type"] for e in sink.events] == [
        "message", "action", "result", "message", "action",
    ]
    assert sink.events[1]["verb"] == "write_file"
    assert sink.events[1]["agent"] == "worker"
    assert sink.events[2]["status"] == "ok"


async def test_agent_emits_parse_error(tmp_path):
    sink = RecordingSink()
    registry = ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)
    agent = Agent(
        client=FakeClient(["::action\n::end", "::action done\n::end"]),
        registry=registry, model="m", system_prompt="s", max_steps=5, sink=sink,
    )
    await agent.run("t")
    assert any(e["type"] == "parse_error" for e in sink.events)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py -v`
Expected: FAIL — `Agent.__init__` got an unexpected keyword argument `sink`.

- [ ] **Step 3: Edit `orchestrator/agent.py`**

Add the import (after the existing protocol import on line 8):
```python
from orchestrator.events import NullSink, make_event, preview
```

Change the `__init__` signature and body. Replace:
```python
    def __init__(
        self,
        client,
        registry,  # any object with execute(action) -> (status, message)
        model: str,
        system_prompt: str,
        max_steps: int = 50,
        terminal_verbs: set[str] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.terminal_verbs = terminal_verbs or {"done"}
```
with:
```python
    def __init__(
        self,
        client,
        registry,  # any object with execute(action) -> (status, message)
        model: str,
        system_prompt: str,
        max_steps: int = 50,
        terminal_verbs: set[str] | None = None,
        sink=None,
        agent_label: str = "agent",
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.terminal_verbs = terminal_verbs or {"done"}
        self.sink = sink or NullSink()
        self.agent_label = agent_label
```

In `run()`, after `messages.append({"role": "assistant", "content": reply})`, add:
```python
            self.sink.emit(make_event(
                "message", agent=self.agent_label, text=preview(reply)
            ))
```

In the `except ProtocolError as exc:` block, add as its first line (before the `messages.append(...)`):
```python
                self.sink.emit(make_event(
                    "parse_error", agent=self.agent_label, error=str(exc)
                ))
```

After `if action is None:` / `break` and BEFORE the `if action.verb in self.terminal_verbs:` check, add:
```python
            self.sink.emit(make_event(
                "action", agent=self.agent_label, verb=action.verb,
                args=action.args, body_preview=preview(action.body),
            ))
```

After `status, message = result` and BEFORE `if status == "stop":`, add:
```python
            self.sink.emit(make_event(
                "result", agent=self.agent_label, status=status,
                message_preview=preview(message),
            ))
```

- [ ] **Step 4: Run the agent tests to verify they pass (including the old ones)**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py -v`
Expected: PASS — the two new tests plus all pre-existing agent tests (backward compat: no-sink agents still behave identically).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/agent.py tests/test_agent.py
git commit -m "feat: emit turn-level events from the agent loop (optional sink)"
```

---

## Task 3: Instrument coordination + orchestrator

**Files:**
- Modify: `orchestrator/coordination.py`
- Modify: `orchestrator/orchestrator.py`
- Test: `tests/test_coordination.py` (append), `tests/test_orchestrator.py` (append)

**Interfaces:**
- Consumes: `make_event`, `plan_event`, `preview`, `NullSink`; `Agent(..., sink=, agent_label=)` from Task 2.
- Produces: `CoordinationRegistry(plan, worker_factory, no_progress_limit=5, sink=None)` emitting `worker_started`, `worker_finished`, `plan`, `no_progress`; `Orchestrator(..., sink=None)` emitting the seed `plan` and wiring the sink into the dominant `Agent` + the `CoordinationRegistry`.

- [ ] **Step 1: Append the failing test to `tests/test_coordination.py`**

```python
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
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_coordination.py::test_coordination_emits_worker_and_plan_events -v`
Expected: FAIL — `CoordinationRegistry.__init__` got an unexpected keyword argument `sink`.

- [ ] **Step 3: Edit `orchestrator/coordination.py`**

Add the import (after the existing `from orchestrator.protocol import Action` line):
```python
from orchestrator.events import NullSink, make_event, plan_event, preview
```

Change `__init__` to accept and store the sink. Replace:
```python
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
```
with:
```python
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
```

Replace the `execute` method body to emit a `plan` event on change and `no_progress` on the backstop. Replace:
```python
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
```
with:
```python
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
```

In `_delegate`, after `self.plan.mark_in_progress(index)` (inside the `try`/`except` it is already guarded) and BEFORE `worker = self.worker_factory()`, add:
```python
        self.sink.emit(make_event(
            "worker_started", step=index, subtask=preview(subtask)
        ))
```
After `result = await worker.run(subtask)` and BEFORE `report = _last_assistant(...)`, add:
```python
        self.sink.emit(make_event(
            "worker_finished", step=index, stopped_reason=result.stopped_reason
        ))
```

- [ ] **Step 4: Run the coordination tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_coordination.py -v`
Expected: PASS — the new test plus all pre-existing coordination tests.

- [ ] **Step 5: Append the failing integration test to `tests/test_orchestrator.py`**

```python
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
```

- [ ] **Step 6: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py::test_orchestrator_emits_full_event_stream -v`
Expected: FAIL — `Orchestrator.__init__` got an unexpected keyword argument `sink`.

- [ ] **Step 7: Edit `orchestrator/orchestrator.py`**

Add the import (after `from orchestrator.plan import Plan`):
```python
from orchestrator.events import NullSink, plan_event
```

Add `sink=None` to `Orchestrator.__init__`. Replace:
```python
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
```
with:
```python
    def __init__(
        self,
        planner,
        worker_factory: Callable[[], Agent],
        dominant_client,
        dominant_model: str,
        max_dominant_turns: int = 40,
        no_progress_limit: int = 5,
        sink=None,
    ) -> None:
        self.planner = planner
        self.worker_factory = worker_factory
        self.dominant_client = dominant_client
        self.dominant_model = dominant_model
        self.max_dominant_turns = max_dominant_turns
        self.no_progress_limit = no_progress_limit
        self.sink = sink or NullSink()
```

In `run()`, after `plan = Plan.from_descriptions(steps)`, add:
```python
        self.sink.emit(plan_event(plan))
```
Change the `CoordinationRegistry(...)` construction to pass the sink. Replace:
```python
        coord = CoordinationRegistry(
            plan, self.worker_factory, no_progress_limit=self.no_progress_limit
        )
```
with:
```python
        coord = CoordinationRegistry(
            plan, self.worker_factory,
            no_progress_limit=self.no_progress_limit, sink=self.sink,
        )
```
Change the dominant `Agent(...)` construction to pass the sink + label. Replace:
```python
        dominant = Agent(
            client=self.dominant_client,
            registry=coord,
            model=self.dominant_model,
            system_prompt=DOMINANT_PROMPT,
            max_steps=self.max_dominant_turns,
            terminal_verbs={"task_complete"},
        )
```
with:
```python
        dominant = Agent(
            client=self.dominant_client,
            registry=coord,
            model=self.dominant_model,
            system_prompt=DOMINANT_PROMPT,
            max_steps=self.max_dominant_turns,
            terminal_verbs={"task_complete"},
            sink=self.sink,
            agent_label="dominant",
        )
```

> Note: `run_started`/`run_finished`/`run_aborted` are emitted by the **server** (Task 5), not the orchestrator, because only the server knows the model ids + research flag and owns the task lifecycle. The orchestrator emits only the seed `plan`; coordination emits subsequent `plan` changes.

- [ ] **Step 8: Run the orchestrator tests, then the full suite**

Run: `.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py -v`
Expected: PASS (new test + existing).
Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS, 0 failures (backward-compat gate — all prior tests still green).

- [ ] **Step 9: Commit**

```bash
git add orchestrator/coordination.py orchestrator/orchestrator.py tests/test_coordination.py tests/test_orchestrator.py
git commit -m "feat: emit worker/plan events from coordination + orchestrator"
```

---

## Task 4: Run lifecycle manager (`run_manager.py`) + config

**Files:**
- Modify: `orchestrator/config.py`
- Create: `orchestrator/run_manager.py`
- Test: `tests/test_config.py` (append), `tests/test_run_manager.py`

**Interfaces:**
- Consumes: `EventBus`, `make_event` from `orchestrator.events`.
- Produces: `RunManager` (`bus`, `is_running`, `start(factory)`, `async stop()`); `Config.host`, `Config.port`. `factory` is `async def (bus: EventBus) -> None`; `_guard` emits `run_aborted` on cancel and `error` on exception.

- [ ] **Step 1: Append the failing config test to `tests/test_config.py`**

```python
def test_server_host_port_defaults():
    cfg = Config()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py::test_server_host_port_defaults -v`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'host'`.

- [ ] **Step 3: Add the fields to `orchestrator/config.py`**

After the Phase 3 fields (`research_timeout: float = 180.0`) and before `__post_init__`, add:
```python
    # Phase 4: web server
    host: str = "127.0.0.1"
    port: int = 8000
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing `RunManager` tests**

`tests/test_run_manager.py`:
```python
import asyncio

from orchestrator.events import make_event
from orchestrator.run_manager import RunManager


async def test_start_runs_factory_to_completion():
    mgr = RunManager()

    async def factory(bus):
        bus.emit(make_event("hello"))

    mgr.start(factory)
    await mgr.task  # wait for completion
    assert mgr.is_running is False
    assert [e["type"] for e in mgr.bus.replay()] == ["hello"]


async def test_start_resets_buffer():
    mgr = RunManager()
    mgr.bus.emit(make_event("stale"))

    async def factory(bus):
        bus.emit(make_event("fresh"))

    mgr.start(factory)
    await mgr.task
    assert [e["type"] for e in mgr.bus.replay()] == ["fresh"]


async def test_stop_emits_run_aborted():
    mgr = RunManager()

    async def factory(bus):
        await asyncio.sleep(10)

    mgr.start(factory)
    await asyncio.sleep(0)  # let the task start
    await mgr.stop()
    assert mgr.is_running is False
    assert any(e["type"] == "run_aborted" for e in mgr.bus.replay())


async def test_factory_exception_emits_error():
    mgr = RunManager()

    async def factory(bus):
        raise RuntimeError("boom")

    mgr.start(factory)
    await mgr.task
    err = [e for e in mgr.bus.replay() if e["type"] == "error"]
    assert err and "boom" in err[0]["message"]
```

- [ ] **Step 6: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_run_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.run_manager'`.

- [ ] **Step 7: Write `orchestrator/run_manager.py`**

```python
"""Owns the single active run: an asyncio task wrapping a run-factory, plus the
EventBus that streams its events. Hard-cancel kills the task; cleanup and the
terminal event live here so the server stays thin."""
from __future__ import annotations

import asyncio
import contextlib

from orchestrator.events import EventBus, make_event


class RunManager:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, factory) -> None:
        """factory: async def (bus: EventBus) -> None. Resets the bus buffer and
        launches the run as a background task."""
        self.bus.reset()
        self.task = asyncio.create_task(self._guard(factory))

    async def _guard(self, factory) -> None:
        try:
            await factory(self.bus)
        except asyncio.CancelledError:
            self.bus.emit(make_event("run_aborted"))
            raise
        except Exception as exc:  # surface, don't crash the server
            self.bus.emit(make_event("error", message=str(exc)))

    async def stop(self) -> None:
        if self.is_running:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_run_manager.py tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add orchestrator/config.py orchestrator/run_manager.py tests/test_config.py tests/test_run_manager.py
git commit -m "feat: add RunManager run lifecycle + server host/port config"
```

---

## Task 5: FastAPI server (`server.py`) + deps + entry point

**Files:**
- Modify: `pyproject.toml`
- Create: `orchestrator/server.py`
- Create: `orchestrator/static/index.html` (placeholder so StaticFiles mounts; replaced in Task 6)
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `RunManager`, `EventBus`, `make_event` (events/run_manager); `Config`; `LMStudioClient`, `mcp_integrations`, `McpResearcher`, `Orchestrator`, `WORKER_PROMPT`, `RESEARCH_HINT`, `LocalPlanner`, `GeminiPlanner`, `ToolRegistry`, `CompositeRegistry`, `Sandbox`, `Agent`, `load_dotenv`.
- Produces: FastAPI `app`; module globals `manager: RunManager`, `cfg: Config`; `build_run_factory(params: dict, cfg: Config)` (monkeypatch seam); `main()` entry point. Routes: `GET /api/models`, `POST /api/run`, `POST /api/stop`, `WS /ws`, `GET /`, mounted `/static`.

- [ ] **Step 1: Add dependencies and install them**

In `pyproject.toml`, change:
```toml
dependencies = ["httpx>=0.27"]
```
to:
```toml
dependencies = ["httpx>=0.27", "fastapi>=0.110", "uvicorn[standard]>=0.29"]
```

Install into the venv:
Run: `.venv\Scripts\python.exe -m pip install -e ".[dev]"`
Expected: installs fastapi + uvicorn (and re-installs the editable package) without error.

- [ ] **Step 2: Create the static placeholder so the app can mount StaticFiles**

`orchestrator/static/index.html` (one line; replaced in Task 6):
```html
<!doctype html><title>orchestrator</title><p>placeholder</p>
```

- [ ] **Step 3: Write the failing server tests**

`tests/test_server.py`:
```python
from fastapi.testclient import TestClient

from orchestrator import server
from orchestrator.events import make_event
from orchestrator.run_manager import RunManager

_BODY = {"dominant": "d", "worker": "w", "project": "./scratch", "goal": "g"}


def test_api_models(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def list_models(self):
            return ["m1", "m2"]

        async def aclose(self):
            pass

    monkeypatch.setattr(server, "LMStudioClient", FakeClient)
    monkeypatch.setattr(server, "mcp_integrations", lambda path: [])
    monkeypatch.delenv("LMSTUDIO_TOKEN", raising=False)
    client = TestClient(server.app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["models"] == ["m1", "m2"]
    assert body["research_available"] is False


def test_run_streams_events_over_ws(monkeypatch):
    server.manager = RunManager()

    async def fake_factory(bus):
        bus.emit(make_event("run_started", goal="g", dominant="d", worker="w", research=False))
        bus.emit(make_event("run_finished", stopped_reason="task_complete"))

    monkeypatch.setattr(server, "build_run_factory", lambda params, cfg: fake_factory)
    client = TestClient(server.app)
    with client.websocket_connect("/ws") as ws:
        assert client.post("/api/run", json=_BODY).status_code == 200
        received = [ws.receive_json()["type"] for _ in range(2)]
    assert "run_started" in received
    assert "run_finished" in received


def test_double_run_returns_409(monkeypatch):
    import asyncio

    server.manager = RunManager()

    async def long_factory(bus):
        await asyncio.sleep(5)

    monkeypatch.setattr(server, "build_run_factory", lambda params, cfg: long_factory)
    client = TestClient(server.app)
    assert client.post("/api/run", json=_BODY).status_code == 200
    assert client.post("/api/run", json=_BODY).status_code == 409
    client.post("/api/stop")


def test_stop_emits_run_aborted(monkeypatch):
    import asyncio

    server.manager = RunManager()

    async def long_factory(bus):
        await asyncio.sleep(5)

    monkeypatch.setattr(server, "build_run_factory", lambda params, cfg: long_factory)
    client = TestClient(server.app)
    client.post("/api/run", json=_BODY)
    assert client.post("/api/stop").status_code == 200
    assert any(e["type"] == "run_aborted" for e in server.manager.bus.replay())
```

- [ ] **Step 4: Run them to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.server'`.

- [ ] **Step 5: Write `orchestrator/server.py`**

```python
"""FastAPI dashboard over the autonomous engine.

A single RunManager owns the one active run (its asyncio task + EventBus). REST
configures/starts/stops a run; the /ws WebSocket replays the current run's
buffered events then streams live ones. build_run_factory is the production run;
it is a module attribute so tests can monkeypatch it."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orchestrator.agent import Agent
from orchestrator.composite_registry import CompositeRegistry
from orchestrator.config import Config
from orchestrator.env import load_dotenv
from orchestrator.events import make_event
from orchestrator.llm_client import LMStudioClient
from orchestrator.mcp_research import McpResearcher, mcp_integrations
from orchestrator.orchestrator import RESEARCH_HINT, WORKER_PROMPT, Orchestrator
from orchestrator.planner import GeminiPlanner, LocalPlanner
from orchestrator.run_manager import RunManager
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
cfg = Config()
manager = RunManager()


class RunParams(BaseModel):
    dominant: str
    worker: str
    project: str
    goal: str
    enable_research: bool = False


def _build_planner(cfg: Config, client: LMStudioClient, dominant_model: str):
    if cfg.planner == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            return GeminiPlanner(api_key=key, model=cfg.gemini_model)
        if not cfg.planner_fallback_local:
            raise RuntimeError("GEMINI_API_KEY not set and fallback disabled")
    return LocalPlanner(client=client, model=dominant_model)


def build_run_factory(params: dict, cfg: Config):
    """Return an async run(bus) coroutine that wires the engine and runs it,
    emitting run_started/run_finished and closing clients in finally."""

    async def _run(bus) -> None:
        token = os.environ.get("LMSTUDIO_TOKEN", "")
        client = LMStudioClient(
            base_url=cfg.lm_studio_url, timeout=cfg.request_timeout, token=token
        )
        researcher = None
        try:
            integrations = mcp_integrations(cfg.resolved_mcp_json())
            research_on = bool(params.get("enable_research")) and bool(token) and bool(integrations)
            worker_prompt = WORKER_PROMPT
            if research_on:
                researcher = McpResearcher(
                    base_url=cfg.lmstudio_native_url, token=token,
                    model=cfg.research_model or params["worker"],
                    integrations=integrations, timeout=cfg.research_timeout,
                )
                worker_prompt = WORKER_PROMPT + RESEARCH_HINT

            bus.emit(make_event(
                "run_started", goal=params["goal"],
                dominant=params["dominant"], worker=params["worker"],
                research=research_on,
            ))

            project = Path(params["project"])
            project.mkdir(parents=True, exist_ok=True)

            def worker_factory() -> Agent:
                tool_registry = ToolRegistry(Sandbox(project), cfg.command_timeout)
                registry = (
                    CompositeRegistry(tool_registry, researcher)
                    if researcher is not None
                    else tool_registry
                )
                return Agent(
                    client=client, registry=registry, model=params["worker"],
                    system_prompt=worker_prompt, max_steps=cfg.max_steps,
                    sink=bus, agent_label="worker",
                )

            planner = _build_planner(cfg, client, params["dominant"])
            orch = Orchestrator(
                planner=planner, worker_factory=worker_factory,
                dominant_client=client, dominant_model=params["dominant"],
                max_dominant_turns=cfg.max_dominant_turns,
                no_progress_limit=cfg.no_progress_limit, sink=bus,
            )
            result = await orch.run(params["goal"])
            bus.emit(make_event("run_finished", stopped_reason=result.stopped_reason))
        finally:
            await client.aclose()
            if researcher is not None:
                await researcher.aclose()

    return _run


@app.get("/api/models")
async def api_models():
    token = os.environ.get("LMSTUDIO_TOKEN", "")
    client = LMStudioClient(
        base_url=cfg.lm_studio_url, timeout=cfg.request_timeout, token=token
    )
    try:
        models = await client.list_models()
    finally:
        await client.aclose()
    research_available = bool(token) and bool(mcp_integrations(cfg.resolved_mcp_json()))
    return {"models": models, "research_available": research_available}


@app.post("/api/run")
async def api_run(params: RunParams):
    if manager.is_running:
        raise HTTPException(status_code=409, detail="a run is already active")
    manager.start(build_run_factory(params.model_dump(), cfg))
    return {"status": "started"}


@app.post("/api/stop")
async def api_stop():
    await manager.stop()
    return {"status": "stopped"}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = manager.bus.subscribe()
    try:
        for event in manager.bus.replay():
            await websocket.send_json(event)
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        manager.bus.unsubscribe(queue)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    load_dotenv()
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the server tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_server.py -v`
Expected: PASS (4 passed).

> If `test_run_streams_events_over_ws` is flaky on a slow machine, it is a test-timing issue, not a code bug — the subscriber is registered before `POST /api/run`, so events are queued. Re-run once; if persistent, the loop is delivering correctly and the assertion set (membership, not exact order) already tolerates ordering.

- [ ] **Step 7: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml orchestrator/server.py orchestrator/static/index.html tests/test_server.py
git commit -m "feat: add FastAPI server (models, run, stop, ws) + RunManager wiring"
```

---

## Task 6: Vanilla SPA (`static/`) + manual verification

**Files:**
- Modify: `orchestrator/static/index.html` (replace placeholder)
- Create: `orchestrator/static/style.css`
- Create: `orchestrator/static/app.js`

**Interfaces:**
- Consumes: `GET /api/models`, `POST /api/run`, `POST /api/stop`, `WS /ws`, and the event types from Task 1–5 (`run_started`, `plan`, `message`, `action`, `result`, `parse_error`, `worker_started`, `worker_finished`, `no_progress`, `run_finished`, `run_aborted`, `error`).

> This task is UI; it has no unit test (the server's behavior is already covered). It ends with a manual launch verification and a final full-suite run.

- [ ] **Step 1: Write `orchestrator/static/index.html`** (replace the placeholder)

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dual-GPU Orchestrator</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header>
    <h1>Dual-GPU Orchestrator</h1>
    <span id="status" class="status idle">idle</span>
  </header>

  <section id="config">
    <div class="row">
      <label>Dominant <select id="dominant"></select></label>
      <label>Worker <select id="worker"></select></label>
    </div>
    <label>Project folder <input id="project" value="./scratch" /></label>
    <label>Goal <textarea id="goal" rows="2" placeholder="What should the engine accomplish?"></textarea></label>
    <div class="row">
      <label><input type="checkbox" id="research" /> <span id="research-label">research (unavailable)</span></label>
      <button id="start">Start</button>
      <button id="kill" disabled>Kill</button>
    </div>
  </section>

  <main>
    <section id="plan-panel">
      <h2>Plan <span id="plan-count"></span></h2>
      <ol id="plan"></ol>
    </section>
    <section id="log-panel">
      <h2>Activity</h2>
      <div id="log"></div>
    </section>
  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `orchestrator/static/style.css`**

```css
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.4 system-ui, sans-serif; color: #e6e6e6; background: #16181d; }
header { display: flex; align-items: center; gap: 12px; padding: 10px 16px; background: #1f2430; }
h1 { font-size: 16px; margin: 0; }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: #8a93a6; margin: 0 0 8px; }
.status { font-size: 12px; padding: 2px 8px; border-radius: 10px; }
.status.idle { background: #333; }
.status.running { background: #1d6f42; }
.status.done { background: #2a4d8f; }
.status.aborted, .status.error { background: #8f2a2a; }
#config { display: flex; flex-direction: column; gap: 8px; padding: 12px 16px; background: #1b1f28; }
#config .row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #aab2c0; }
#config .row label { flex-direction: row; align-items: center; gap: 6px; }
input, textarea, select { background: #0f1116; color: #e6e6e6; border: 1px solid #333a47; border-radius: 4px; padding: 6px; font: inherit; }
button { background: #2a4d8f; color: #fff; border: 0; border-radius: 4px; padding: 8px 16px; cursor: pointer; }
button:disabled { background: #333a47; color: #777; cursor: default; }
#kill:not(:disabled) { background: #8f2a2a; }
main { display: grid; grid-template-columns: 320px 1fr; gap: 12px; padding: 12px 16px; }
#plan { list-style: none; padding: 0; margin: 0; }
#plan li { padding: 6px 8px; border-left: 3px solid #333a47; margin-bottom: 4px; background: #1b1f28; }
#plan li.in_progress { border-color: #c9a227; }
#plan li.done { border-color: #1d6f42; opacity: .7; }
.entry { padding: 6px 8px; margin-bottom: 4px; border-radius: 4px; background: #1b1f28; white-space: pre-wrap; word-break: break-word; }
.entry .who { font-weight: 600; margin-right: 6px; }
.entry.dominant .who { color: #6fa8ff; }
.entry.worker .who { color: #6fd3a8; }
.entry.action { background: #232a1b; }
.entry.result { background: #1b2228; }
.entry.system { color: #c9a227; font-style: italic; }
.entry.error { background: #2e1b1b; color: #ff9b9b; }
```

- [ ] **Step 3: Write `orchestrator/static/app.js`**

```javascript
"use strict";

const $ = (id) => document.getElementById(id);
const log = $("log");

function setStatus(text, cls) {
  const el = $("status");
  el.textContent = text;
  el.className = "status " + cls;
}

function addEntry(cls, who, text) {
  const div = document.createElement("div");
  div.className = "entry " + cls;
  if (who) {
    const span = document.createElement("span");
    span.className = "who";
    span.textContent = who;
    div.appendChild(span);
  }
  div.appendChild(document.createTextNode(text));
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

function renderPlan(ev) {
  $("plan-count").textContent = `(${ev.done}/${ev.total})`;
  const ol = $("plan");
  ol.innerHTML = "";
  for (const step of ev.steps) {
    const li = document.createElement("li");
    li.className = step.status;
    li.textContent = `${step.index}. ${step.description}`;
    ol.appendChild(li);
  }
}

function running(isRunning) {
  $("start").disabled = isRunning;
  $("kill").disabled = !isRunning;
}

const handlers = {
  run_started: (e) => { running(true); setStatus("running", "running"); addEntry("system", "", `▶ run started: ${e.goal}`); },
  plan: (e) => renderPlan(e),
  message: (e) => addEntry(e.agent, e.agent + ":", e.text),
  action: (e) => addEntry("action " + e.agent, e.agent + " ⇒ " + e.verb, JSON.stringify(e.args) + (e.body_preview ? "\n" + e.body_preview : "")),
  result: (e) => addEntry("result", "result " + e.status, e.message_preview),
  parse_error: (e) => addEntry("error", e.agent + " parse error", e.error),
  worker_started: (e) => addEntry("system", "", `→ delegating step ${e.step}: ${e.subtask}`),
  worker_finished: (e) => addEntry("system", "", `← worker finished step ${e.step} (${e.stopped_reason})`),
  no_progress: () => addEntry("system", "", "⚠ no progress — stopping"),
  run_finished: (e) => { running(false); setStatus("finished: " + e.stopped_reason, "done"); addEntry("system", "", "■ run finished: " + e.stopped_reason); },
  run_aborted: () => { running(false); setStatus("aborted", "aborted"); addEntry("system", "", "■ run aborted"); },
  error: (e) => { running(false); setStatus("error", "error"); addEntry("error", "error", e.message); },
};

function connectWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    (handlers[ev.type] || (() => {}))(ev);
  };
  ws.onclose = () => setTimeout(connectWs, 1000);
}

async function loadModels() {
  const resp = await fetch("/api/models");
  if (!resp.ok) { addEntry("error", "error", "could not reach LM Studio via /api/models"); return; }
  const data = await resp.json();
  for (const id of ["dominant", "worker"]) {
    const sel = $(id);
    sel.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m; opt.textContent = m;
      sel.appendChild(opt);
    }
  }
  if (data.models.length > 1) $("worker").selectedIndex = 1;
  const cb = $("research");
  cb.disabled = !data.research_available;
  cb.checked = data.research_available;
  $("research-label").textContent = data.research_available ? "research (available)" : "research (unavailable)";
}

$("start").onclick = async () => {
  log.innerHTML = "";
  $("plan").innerHTML = "";
  const body = {
    dominant: $("dominant").value,
    worker: $("worker").value,
    project: $("project").value,
    goal: $("goal").value,
    enable_research: $("research").checked,
  };
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (resp.status === 409) addEntry("error", "error", "a run is already active");
};

$("kill").onclick = () => fetch("/api/stop", { method: "POST" });

connectWs();
loadModels();
```

- [ ] **Step 4: Run the full test suite (nothing should have broken)**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS, 0 failures.

- [ ] **Step 5: Manual launch verification**

Run: `.venv\Scripts\python.exe -m orchestrator.server`
Expected: uvicorn starts on `http://127.0.0.1:8000`. Open it in a browser:
- The dominant/worker dropdowns populate from `/api/models` (LM Studio must be running with models loaded).
- Enter a trivial goal (e.g. `create hello.txt containing "hi"`) with project `./scratch`, click **Start**.
- The Activity log streams dominant/worker messages, actions, and results; the Plan panel updates step statuses live; status pill shows `running` then `finished: task_complete`.
- Start another run and click **Kill** mid-run; status flips to `aborted` and the log shows the abort. Stop the server with Ctrl+C.

- [ ] **Step 6: Commit**

```bash
git add orchestrator/static/index.html orchestrator/static/style.css orchestrator/static/app.js
git commit -m "feat: add vanilla single-page dashboard (live log, plan, kill switch)"
```

---

## Done criteria for Phase 4

- Full suite green (0 failures), including all pre-existing tests (backward-compat gate: no-sink agents/orchestrator behave exactly as before).
- `python -m orchestrator.server` serves a dashboard that auto-fills models, starts an autonomous run, streams turn-level events live (conversation, actions, results, plan checklist), and kills a run instantly.
- The event sink is threaded through `agent`/`coordination`/`orchestrator` with `NullSink` defaults; `cli.py` is unchanged and still runs headless.
- One run at a time (second `POST /api/run` → 409); kill = hard cancel with clean client shutdown and a single `run_aborted`.
- No frontend build step or CDN; `llm_client.py` untouched (`stream: False`).

**Next phase (not in scope):** run history/persistence; per-tool research verbs; multi-run.
```
