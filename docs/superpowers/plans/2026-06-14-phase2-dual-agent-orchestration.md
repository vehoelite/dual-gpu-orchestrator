# Phase 2: Dual-Agent Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the engine autonomous and dual-agent: a pluggable Planner turns a goal into a checklist, a Dominant (9B) orchestrator drives it by delegating each step to a fresh Worker (4B), reviewing results, and completing the run — fully headless.

**Architecture:** Reuse the Phase 1 `Agent` loop for BOTH agents. The worker = `Agent` + `ToolRegistry` (Phase 1, unchanged). The dominant = the same `Agent` loop + a new async `CoordinationRegistry` (set_plan/delegate/mark_done/revise_plan; task_complete is a terminal verb). `Orchestrator.run(goal)` calls the planner, seeds a `Plan`, runs the dominant, and enforces max-turns / no-progress backstops. `delegate` spins up a fresh worker per subtask (no dominant-history leak).

**Tech Stack:** Python 3.11+, `httpx` (async), `pytest` + `pytest-asyncio`. Gemini via `httpx` (no SDK). No native tool-calling. Builds on the merged Phase 1 `orchestrator/` package.

**Reference spec:** `docs/superpowers/specs/2026-06-14-phase2-dual-agent-orchestration-design.md`

---

## File Structure

```
orchestrator/
  plan.py            # NEW: Step, Plan (transitions/render/signature), PlanError, parse_checklist
  agent.py           # MODIFY: terminal_verbs, await-tolerant execute, "stop" status
  planner.py         # NEW: Planner protocol, LocalPlanner, GeminiPlanner
  coordination.py    # NEW: CoordinationRegistry (dominant verbs + no-progress)
  orchestrator.py    # NEW: Orchestrator.run -> RunResult, DOMINANT_PROMPT, WORKER_PROMPT
  config.py          # MODIFY: Phase 2 settings
  cli.py             # NEW: headless run entry point (python -m orchestrator.cli)
tests/
  test_plan.py         # NEW
  test_agent.py        # MODIFY (append 3 tests)
  test_planner.py      # NEW
  test_coordination.py # NEW
  test_orchestrator.py # NEW
  test_config.py       # MODIFY (append 1 test)
```

**Shared interfaces (define once, used everywhere — do not rename):**
- `Step(description: str, status: str = "pending")` — status ∈ `pending|in_progress|done`.
- `Plan` with `from_descriptions(list[str]) -> Plan`, `mark_in_progress(i)`, `mark_done(i)`, `revise(list[str])`, `all_done() -> bool`, `signature() -> tuple`, `render() -> str`; bad index raises `PlanError`.
- `parse_checklist(text: str) -> list[str]` (in `plan.py`).
- `Planner` protocol: `async make_plan(goal: str) -> list[str]`. Impls `LocalPlanner(client, model)`, `GeminiPlanner(api_key, model="gemini-2.0-flash", http_client=None, timeout=60.0)`.
- `CoordinationRegistry(plan: Plan, worker_factory: Callable[[], Agent], no_progress_limit: int = 5)` with `async execute(action) -> tuple[str, str]`; collects `worker_results: list[dict]`.
- `Agent(client, registry, model, system_prompt, max_steps=50, terminal_verbs: set[str] | None = None)` — `terminal_verbs` defaults to `{"done"}`; `registry.execute` may be sync or async; a `("stop", reason)` result ends the loop with `stopped_reason=reason`.
- `Orchestrator(planner, worker_factory, dominant_client, dominant_model, max_dominant_turns=40, no_progress_limit=5)` with `async run(goal) -> RunResult`.
- `RunResult(plan, dominant_transcript, worker_results, stopped_reason)` — `stopped_reason` ∈ `task_complete|max_turns|no_progress|no_action|planner_failed`.

All pytest run via `.venv\Scripts\python.exe -m pytest` (PowerShell, Windows). `asyncio_mode = "auto"` (async tests need no decorator). Set local git identity if needed: `git config user.email "vehoelite@gmail.com"; git config user.name "Jacob"`.

---

## Task 1: Plan model (`plan.py`)

**Files:**
- Create: `orchestrator/plan.py`
- Test: `tests/test_plan.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_plan.py`:
```python
import pytest

from orchestrator.plan import Plan, PlanError, Step, parse_checklist


def test_from_descriptions_all_pending():
    plan = Plan.from_descriptions(["a", "b"])
    assert [s.status for s in plan.steps] == ["pending", "pending"]
    assert plan.steps[0].description == "a"


def test_mark_in_progress_and_done():
    plan = Plan.from_descriptions(["a", "b"])
    plan.mark_in_progress(0)
    assert plan.steps[0].status == "in_progress"
    plan.mark_done(0)
    assert plan.steps[0].status == "done"


def test_bad_index_raises():
    plan = Plan.from_descriptions(["a"])
    with pytest.raises(PlanError):
        plan.mark_done(5)


def test_revise_replaces_steps():
    plan = Plan.from_descriptions(["a", "b"])
    plan.mark_done(0)
    plan.revise(["x", "y", "z"])
    assert [s.description for s in plan.steps] == ["x", "y", "z"]
    assert all(s.status == "pending" for s in plan.steps)


def test_all_done():
    plan = Plan.from_descriptions(["a", "b"])
    assert not plan.all_done()
    plan.mark_done(0)
    plan.mark_done(1)
    assert plan.all_done()


def test_empty_plan_not_all_done():
    assert not Plan().all_done()


def test_signature_changes_with_status():
    plan = Plan.from_descriptions(["a"])
    sig1 = plan.signature()
    plan.mark_done(0)
    assert plan.signature() != sig1


def test_render_contains_status_and_index():
    plan = Plan.from_descriptions(["write code"])
    plan.mark_done(0)
    out = plan.render()
    assert "1/1 done" in out
    assert "[done] 0. write code" in out


def test_parse_checklist_numbered():
    assert parse_checklist("1. do a\n2. do b\n3. do c") == ["do a", "do b", "do c"]


def test_parse_checklist_bullets_and_noise():
    text = "Here is the plan:\n- alpha\n* beta\n\nThanks!"
    assert parse_checklist(text) == ["alpha", "beta"]


def test_parse_checklist_paren_numbers():
    assert parse_checklist("1) first\n2) second") == ["first", "second"]


def test_parse_checklist_empty():
    assert parse_checklist("no list here") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_plan.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.plan'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/plan.py`:
```python
"""The dominant's checklist: ordered steps with status, plus transitions.

Also hosts ``parse_checklist`` since turning checklist text into steps is part
of building a Plan (used by the planner and the coordination verbs)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


class PlanError(Exception):
    """Raised on an invalid plan operation (e.g. a bad step index)."""


@dataclass
class Step:
    description: str
    status: str = "pending"  # "pending" | "in_progress" | "done"


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_descriptions(cls, descriptions: list[str]) -> "Plan":
        return cls(steps=[Step(d) for d in descriptions])

    def _check(self, index: int) -> None:
        if not 0 <= index < len(self.steps):
            raise PlanError(f"no step {index} (plan has {len(self.steps)} steps)")

    def mark_in_progress(self, index: int) -> None:
        self._check(index)
        self.steps[index].status = "in_progress"

    def mark_done(self, index: int) -> None:
        self._check(index)
        self.steps[index].status = "done"

    def revise(self, descriptions: list[str]) -> None:
        self.steps = [Step(d) for d in descriptions]

    def all_done(self) -> bool:
        return len(self.steps) > 0 and all(s.status == "done" for s in self.steps)

    def signature(self) -> tuple:
        return tuple((s.description, s.status) for s in self.steps)

    def render(self) -> str:
        done = sum(1 for s in self.steps if s.status == "done")
        lines = [f"Plan ({done}/{len(self.steps)} done):"]
        for i, s in enumerate(self.steps):
            lines.append(f"[{s.status}] {i}. {s.description}")
        return "\n".join(lines)


def parse_checklist(text: str) -> list[str]:
    """Extract step descriptions from a numbered/bulleted checklist."""
    steps: list[str] = []
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match:
            steps.append(match.group(1).strip())
    return steps
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_plan.py -v`
Expected: PASS (12 passed). Then full suite `.venv\Scripts\python.exe -m pytest -q` → 46 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestrator/plan.py tests/test_plan.py
git commit -m "feat: add Plan model and checklist parsing"
```

---

## Task 2: Extend the Agent loop (`agent.py`)

**Files:**
- Modify: `orchestrator/agent.py`
- Test: `tests/test_agent.py` (append 3 tests)

Three backward-compatible changes: configurable `terminal_verbs`, await-tolerant `registry.execute`, and a `("stop", reason)` result that ends the loop. Default behavior (worker) is unchanged, so all existing agent tests must still pass.

- [ ] **Step 1: Append the failing tests to `tests/test_agent.py`**

Add at the end of `tests/test_agent.py` (the file already defines `FakeClient` and imports `Agent`, `AgentResult`):
```python
class AsyncEchoRegistry:
    """Async registry: records actions, returns ok; 'halt' returns a stop."""

    def __init__(self):
        self.executed = []

    async def execute(self, action):
        self.executed.append(action)
        if action.verb == "halt":
            return ("stop", "no_progress")
        return ("ok", f"did {action.verb}")


async def test_terminal_verb_configurable(tmp_path):
    reg = AsyncEchoRegistry()
    client = FakeClient(["::action task_complete\n::end"])
    agent = Agent(
        client=client, registry=reg, model="m", system_prompt="s",
        max_steps=5, terminal_verbs={"task_complete"},
    )
    result = await agent.run("go")
    assert result.stopped_reason == "task_complete"
    assert reg.executed == []  # terminal verb is not executed by the registry


async def test_async_registry_is_awaited(tmp_path):
    reg = AsyncEchoRegistry()
    client = FakeClient(["::action foo\n::end", "::action done\n::end"])
    agent = Agent(client=client, registry=reg, model="m", system_prompt="s", max_steps=5)
    result = await agent.run("go")
    assert result.stopped_reason == "done"
    assert [a.verb for a in reg.executed] == ["foo"]
    assert any("::result ok" in m["content"] for m in agent.client.calls[1])


async def test_stop_status_ends_run(tmp_path):
    reg = AsyncEchoRegistry()
    client = FakeClient(["::action halt\n::end", "::action done\n::end"])
    agent = Agent(client=client, registry=reg, model="m", system_prompt="s", max_steps=5)
    result = await agent.run("go")
    assert result.stopped_reason == "no_progress"
    assert len(agent.client.calls) == 1
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py -k "terminal_verb_configurable or async_registry_is_awaited or stop_status_ends_run" -v`
Expected: FAIL (e.g. `TypeError: __init__() got an unexpected keyword argument 'terminal_verbs'` and the async/stop tests failing).

- [ ] **Step 3: Apply edit A — imports**

In `orchestrator/agent.py`, replace:
```python
from __future__ import annotations

from dataclasses import dataclass

from orchestrator.protocol import ProtocolError, parse_action, serialize_result
from orchestrator.tools import ToolRegistry
```
with:
```python
from __future__ import annotations

import inspect
from dataclasses import dataclass

from orchestrator.protocol import ProtocolError, parse_action, serialize_result
```
(The `ToolRegistry` import is removed — it was only used in a type hint that becomes generic below.)

- [ ] **Step 4: Apply edit B — constructor**

Replace:
```python
    def __init__(
        self,
        client,
        registry: ToolRegistry,
        model: str,
        system_prompt: str,
        max_steps: int = 50,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps
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
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.terminal_verbs = terminal_verbs or {"done"}
```

- [ ] **Step 5: Apply edit C — loop body**

Replace:
```python
            if action is None:
                reason = "no_action"
                break
            if action.verb == "done":
                reason = "done"
                break

            status, message = self.registry.execute(action)
            messages.append(
                {"role": "user", "content": serialize_result(status, message)}
            )
```
with:
```python
            if action is None:
                reason = "no_action"
                break
            if action.verb in self.terminal_verbs:
                reason = action.verb
                break

            result = self.registry.execute(action)
            if inspect.isawaitable(result):
                result = await result
            status, message = result
            if status == "stop":
                reason = message
                break
            messages.append(
                {"role": "user", "content": serialize_result(status, message)}
            )
```

- [ ] **Step 6: Run the agent suite to verify all pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_agent.py -v`
Expected: PASS (9 passed — 6 existing + 3 new). Then full suite `.venv\Scripts\python.exe -m pytest -q` → 49 passed.

- [ ] **Step 7: Commit**

```bash
git add orchestrator/agent.py tests/test_agent.py
git commit -m "feat: make Agent loop reusable for the dominant (terminal_verbs, async/stop)"
```

---

## Task 3: Planner (`planner.py`)

**Files:**
- Create: `orchestrator/planner.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_planner.py`:
```python
import httpx
import pytest

from orchestrator.planner import GeminiPlanner, LocalPlanner


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []

    async def complete(self, model, messages, temperature=0.7):
        self.calls.append((model, messages))
        return self.text


async def test_local_planner_parses_model_output():
    client = FakeClient("1. write tests\n2. implement")
    planner = LocalPlanner(client=client, model="dom")
    steps = await planner.make_plan("build a thing")
    assert steps == ["write tests", "implement"]
    assert client.calls[0][1][-1]["content"] == "build a thing"


async def test_gemini_planner_request_and_parse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.url.params.get("key")
        return httpx.Response(
            200,
            json={"candidates": [
                {"content": {"parts": [{"text": "1. step one\n2. step two"}]}}
            ]},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    planner = GeminiPlanner(
        api_key="SECRET", model="gemini-2.0-flash", http_client=http_client
    )
    steps = await planner.make_plan("goal")
    assert steps == ["step one", "step two"]
    assert "gemini-2.0-flash:generateContent" in captured["url"]
    assert captured["key"] == "SECRET"


async def test_gemini_planner_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="forbidden")

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    planner = GeminiPlanner(api_key="bad", http_client=http_client)
    with pytest.raises(httpx.HTTPStatusError):
        await planner.make_plan("goal")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_planner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.planner'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/planner.py`:
```python
"""Pluggable planner: goal -> ordered checklist. Frontier (Gemini) or local.

The key insight (see spec): the plan is the highest-leverage artifact, so it can
use a high-quality frontier model while execution stays local."""
from __future__ import annotations

from typing import Protocol

import httpx

from orchestrator.plan import parse_checklist

_PLANNER_SYSTEM = (
    "You are a planning assistant. Break the user's goal into a short, ordered "
    "checklist of concrete, self-contained steps a developer agent can execute "
    "one at a time. Output ONLY the checklist, one step per line, like:\n"
    "1. first step\n2. second step\nNo preamble, no commentary."
)


class Planner(Protocol):
    async def make_plan(self, goal: str) -> list[str]: ...


class LocalPlanner:
    """Plan with a local LM Studio model via the existing LMStudioClient."""

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    async def make_plan(self, goal: str) -> list[str]:
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": goal},
        ]
        text = await self.client.complete(model=self.model, messages=messages)
        return parse_checklist(text)


class GeminiPlanner:
    """Plan with Google's Gemini API. Key is supplied by the caller (env-sourced);
    it is never hard-coded or committed."""

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def make_plan(self, goal: str) -> list[str]:
        url = f"{self._BASE}/{self.model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": _PLANNER_SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": goal}]}],
        }
        resp = await self._client.post(url, params={"key": self.api_key}, json=payload)
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_checklist(text)

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_planner.py -v`
Expected: PASS (3 passed). Then full suite `.venv\Scripts\python.exe -m pytest -q` → 52 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestrator/planner.py tests/test_planner.py
git commit -m "feat: add pluggable planner (local + Gemini)"
```

---

## Task 4: Coordination registry (`coordination.py`)

**Files:**
- Create: `orchestrator/coordination.py`
- Test: `tests/test_coordination.py`

The dominant's verbs over a `Plan`. `execute` is async (so `delegate` can run a worker). It tracks no-progress: if `no_progress_limit` consecutive actions leave the plan signature unchanged, it returns `("stop", "no_progress")`.

- [ ] **Step 1: Write the failing tests**

`tests/test_coordination.py`:
```python
from orchestrator.agent import AgentResult
from orchestrator.coordination import CoordinationRegistry
from orchestrator.plan import Plan
from orchestrator.protocol import Action


class FakeWorker:
    def __init__(self, report="done it", stopped_reason="done"):
        self.report = report
        self.stopped_reason = stopped_reason
        self.received = None

    async def run(self, task):
        self.received = task
        return AgentResult(
            transcript=[{"role": "assistant", "content": self.report}],
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_coordination.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.coordination'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/coordination.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_coordination.py -v`
Expected: PASS (8 passed). Then full suite `.venv\Scripts\python.exe -m pytest -q` → 60 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestrator/coordination.py tests/test_coordination.py
git commit -m "feat: add coordination registry with delegate and no-progress backstop"
```

---

## Task 5: Orchestrator (`orchestrator.py`)

**Files:**
- Create: `orchestrator/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_orchestrator.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.orchestrator'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/orchestrator.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_orchestrator.py -v`
Expected: PASS (4 passed). Then full suite `.venv\Scripts\python.exe -m pytest -q` → 64 passed.

- [ ] **Step 5: Commit**

```bash
git add orchestrator/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add Orchestrator wiring planner, dominant, and backstops"
```

---

## Task 6: Config + headless CLI (`config.py`, `cli.py`)

**Files:**
- Modify: `orchestrator/config.py`
- Create: `orchestrator/cli.py`
- Test: `tests/test_config.py` (append 1 test)

- [ ] **Step 1: Append the failing config test to `tests/test_config.py`**

Add at the end of `tests/test_config.py`:
```python
def test_phase2_defaults():
    cfg = Config()
    assert cfg.planner == "local"
    assert cfg.planner_fallback_local is True
    assert cfg.max_dominant_turns > 0
    assert cfg.no_progress_limit > 0
    assert cfg.gemini_model
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py::test_phase2_defaults -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'planner'`.

- [ ] **Step 3: Add Phase 2 fields to `Config`**

In `orchestrator/config.py`, replace the field block:
```python
@dataclass
class Config:
    lm_studio_url: str = "http://localhost:1234/v1"
    mcp_json_path: str = ""
    request_timeout: float = 120.0
    command_timeout: float = 60.0
    max_steps: int = 50
```
with:
```python
@dataclass
class Config:
    lm_studio_url: str = "http://localhost:1234/v1"
    mcp_json_path: str = ""
    request_timeout: float = 120.0
    command_timeout: float = 60.0
    max_steps: int = 50
    # Phase 2: orchestration
    planner: str = "local"  # "local" | "gemini"
    gemini_model: str = "gemini-2.0-flash"
    planner_fallback_local: bool = True
    max_dominant_turns: int = 40
    no_progress_limit: int = 5
    dominant_model: str = ""
    worker_model: str = ""
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: PASS (3 passed — 2 existing + 1 new).

- [ ] **Step 5: Write the headless CLI**

`orchestrator/cli.py`:
```python
"""Headless autonomous run.

Usage:
    python -m orchestrator.cli "<goal>" <project_folder>

Picks dominant/worker from LM Studio's loaded models (first two). Planner is
chosen by Config.planner; the Gemini key is read from GEMINI_API_KEY (never
committed). Runs to completion with no human input.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from orchestrator.agent import Agent
from orchestrator.config import Config
from orchestrator.llm_client import LMStudioClient
from orchestrator.orchestrator import Orchestrator, WORKER_PROMPT
from orchestrator.planner import GeminiPlanner, LocalPlanner
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


def _build_planner(cfg: Config, client: LMStudioClient):
    if cfg.planner == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            return GeminiPlanner(api_key=key, model=cfg.gemini_model)
        if not cfg.planner_fallback_local:
            raise SystemExit("GEMINI_API_KEY not set and fallback disabled")
        print("GEMINI_API_KEY not set; falling back to local planner")
    return LocalPlanner(client=client, model=cfg.dominant_model)


async def main() -> int:
    if len(sys.argv) < 3:
        print('usage: python -m orchestrator.cli "<goal>" <project_folder>')
        return 2
    goal = sys.argv[1]
    project = Path(sys.argv[2])
    project.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)
    try:
        models = await client.list_models()
        if not models:
            raise SystemExit("no models loaded in LM Studio")
        cfg.dominant_model = cfg.dominant_model or models[0]
        cfg.worker_model = cfg.worker_model or (models[1] if len(models) > 1 else models[0])
        print(f"dominant={cfg.dominant_model} worker={cfg.worker_model}")

        def worker_factory() -> Agent:
            return Agent(
                client=client,
                registry=ToolRegistry(Sandbox(project), cfg.command_timeout),
                model=cfg.worker_model,
                system_prompt=WORKER_PROMPT,
                max_steps=cfg.max_steps,
            )

        planner = _build_planner(cfg, client)
        orch = Orchestrator(
            planner=planner,
            worker_factory=worker_factory,
            dominant_client=client,
            dominant_model=cfg.dominant_model,
            max_dominant_turns=cfg.max_dominant_turns,
            no_progress_limit=cfg.no_progress_limit,
        )
        result = await orch.run(goal)
        print(f"\nStopped: {result.stopped_reason}")
        print(result.plan.render())
    finally:
        await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 6: Verify the CLI shows usage without a server**

Run: `.venv\Scripts\python.exe -m orchestrator.cli`
Expected: prints `usage: python -m orchestrator.cli "<goal>" <project_folder>` and exits with code 2. (Do NOT attempt a real run here — that needs LM Studio and is out of scope for the unit phase.)

- [ ] **Step 7: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS (65 passed).

- [ ] **Step 8: Commit**

```bash
git add orchestrator/config.py orchestrator/cli.py tests/test_config.py
git commit -m "feat: add Phase 2 config and headless run CLI"
```

---

## Done criteria for Phase 2

- Full suite green (65 passed).
- New modules exist with the interfaces named in File Structure; `agent.py` extended backward-compatibly.
- A complete autonomous run works deterministically in tests: goal → planner → dominant delegates → fresh workers execute → `task_complete`, with max-turns and no-progress backstops verified.
- No native tool-calling anywhere; all agent action flows through the text protocol.
- The Gemini key is never hard-coded or committed (env var only).

**Manual live smoke (optional, outside the unit suite):** start LM Studio with two models loaded, then
`.venv\Scripts\python.exe -m orchestrator.cli "create hello.txt containing hello world" ./scratch`
(set `GEMINI_API_KEY` and `Config.planner="gemini"` to exercise the frontier planner).

**Next phases (not in scope here):** Phase 3 = `mcp_host.py` + Exa (`web_search` verb). Phase 4 = FastAPI + WebSocket UI + interactive kill switch.
