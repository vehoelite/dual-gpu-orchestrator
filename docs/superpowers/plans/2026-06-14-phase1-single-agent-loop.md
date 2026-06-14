# Phase 1: Single-Agent Text-Protocol Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single LM-Studio-backed agent that reasons in prose, emits one `::action` block per turn, and has the engine execute first-party filesystem/command tools against a sandboxed project folder — end to end.

**Architecture:** A model talks to LM Studio's OpenAI-compatible `/v1/chat/completions` (plain completions, no `tools` param). Its text output is parsed by `protocol.py` into an `Action`; `tools.py` executes the action via a registry, with all paths contained by `sandbox.py`; the result is serialized back into the conversation as a `::result` block. `agent.py` drives the emit→parse→execute→feed-back loop until the model emits `::action done`, emits no action, or hits a max-step cap.

**Tech Stack:** Python 3.11+, `httpx` (async), `pytest` + `pytest-asyncio`. No native OpenAI tool-calling. Top-level `orchestrator/` package, tests under `tests/`.

**Reference spec:** `docs/superpowers/specs/2026-06-15-dual-gpu-orchestrator-design.md`

---

## File Structure

```
dual-gpu-orchestrator/
  pyproject.toml              # deps + pytest config
  conftest.py                 # empty; puts repo root on sys.path
  orchestrator/
    __init__.py
    config.py                 # Config dataclass (URLs, timeouts, limits)
    protocol.py               # Action, parse_action, serialize_result, ProtocolError
    sandbox.py                # Sandbox.resolve, SandboxError
    tools.py                  # ToolRegistry + read_file/write_file/list_dir/run_command, ToolError
    llm_client.py             # LMStudioClient.list_models / .complete
    agent.py                  # Agent.run -> AgentResult
    smoke.py                  # manual end-to-end check against real LM Studio
  tests/
    test_protocol.py
    test_sandbox.py
    test_tools.py
    test_llm_client.py
    test_agent.py
```

**Shared interfaces (defined once, used everywhere — do not rename):**
- `Action(verb: str, args: dict[str, str], body: str)` — `body` is `""` when absent.
- `parse_action(text: str) -> Action | None` — `None` when no `::action` marker; raises `ProtocolError` when a marker is present but malformed.
- `serialize_result(status: str, message: str) -> str` — `status` in `{"ok", "error"}`.
- `Sandbox(root: Path)` with `.resolve(path: str) -> Path`; raises `SandboxError` on escape.
- `ToolRegistry(sandbox: Sandbox, command_timeout: float)` with `.execute(action: Action) -> tuple[str, str]` returning `(status, message)`; tool handlers raise `ToolError` on failure.
- `LMStudioClient(base_url: str, timeout: float, http_client: httpx.AsyncClient | None = None)` with async `.list_models() -> list[str]` and async `.complete(model, messages, temperature=0.7) -> str`.
- `Agent(client, registry, model, system_prompt, max_steps)` with async `.run(task: str) -> AgentResult`.
- `AgentResult(transcript: list[dict], stopped_reason: str)` — `stopped_reason` in `{"done", "no_action", "max_steps"}`.

---

## Task 0: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `conftest.py`
- Create: `orchestrator/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "orchestrator"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["httpx>=0.27"]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Create `conftest.py` (empty)**

```python
# Empty on purpose: presence makes pytest add the repo root to sys.path,
# so `import orchestrator` works without an editable install.
```

- [ ] **Step 3: Create package markers**

`orchestrator/__init__.py`:
```python
"""Dual-GPU autonomous dual-agent engine."""
```

`tests/__init__.py`:
```python
```

- [ ] **Step 4: Create and activate a venv, install dev deps**

Run (Windows PowerShell):
```
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ".[dev]"
```
Expected: installs `httpx`, `pytest`, `pytest-asyncio` and the `orchestrator` package without error.

- [ ] **Step 5: Verify pytest runs (no tests yet)**

Run: `pytest -q`
Expected: `no tests ran` (exit code 5) — confirms pytest is configured.

- [ ] **Step 6: Add a `.gitignore`**

Create `.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml conftest.py orchestrator/__init__.py tests/__init__.py .gitignore
git commit -m "chore: scaffold orchestrator package and pytest setup"
```

---

## Task 1: Protocol parser (`protocol.py`)

**Files:**
- Create: `orchestrator/protocol.py`
- Test: `tests/test_protocol.py`

**Format being parsed:**
```
::action <verb>
key: value
another: value
---
optional freeform body
::end
```
Rules: leading prose before `::action` is ignored. `key: value` lines (before `---` or `::end`) populate `args`. An optional `---` line starts the body, which runs verbatim until `::end`. `::end` is required. Lines are stripped of surrounding whitespace for the marker/key parsing; body lines are preserved verbatim (only the trailing newline before `::end` is trimmed).

- [ ] **Step 1: Write the failing tests**

`tests/test_protocol.py`:
```python
import pytest

from orchestrator.protocol import (
    Action,
    ProtocolError,
    parse_action,
    serialize_result,
)


def test_no_action_marker_returns_none():
    assert parse_action("I am just thinking out loud.") is None


def test_parse_action_with_args_only():
    text = (
        "Let me read the file.\n"
        "::action read_file\n"
        "path: notes.md\n"
        "::end\n"
    )
    action = parse_action(text)
    assert action == Action(verb="read_file", args={"path": "notes.md"}, body="")


def test_parse_action_with_body():
    text = (
        "::action write_file\n"
        "path: hello.txt\n"
        "---\n"
        "Hello\n"
        "world\n"
        "::end\n"
    )
    action = parse_action(text)
    assert action.verb == "write_file"
    assert action.args == {"path": "hello.txt"}
    assert action.body == "Hello\nworld"


def test_parse_tolerates_surrounding_whitespace():
    text = "  ::action   list_dir  \n   path:   .  \n  ::end  "
    action = parse_action(text)
    assert action.verb == "list_dir"
    assert action.args == {"path": "."}
    assert action.body == ""


def test_missing_end_raises_protocol_error():
    text = "::action read_file\npath: notes.md\n"
    with pytest.raises(ProtocolError):
        parse_action(text)


def test_missing_verb_raises_protocol_error():
    text = "::action\npath: notes.md\n::end\n"
    with pytest.raises(ProtocolError):
        parse_action(text)


def test_serialize_result_ok():
    assert serialize_result("ok", "wrote 5 bytes") == (
        "::result ok\nwrote 5 bytes\n::end"
    )


def test_serialize_result_error():
    assert serialize_result("error", "no such file") == (
        "::result error\nno such file\n::end"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_protocol.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.protocol'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/protocol.py`:
```python
"""Parse the structured text action protocol and serialize results.

Agents act by emitting an ``::action`` block in their normal text output:

    ::action <verb>
    key: value
    ---
    optional body
    ::end

The parser is deliberately forgiving so weak models can use it reliably.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_RE = re.compile(
    r"::action[ \t]+(?P<header>.*?)\n(?P<inner>.*?)\n?[ \t]*::end",
    re.DOTALL,
)


class ProtocolError(Exception):
    """Raised when an ``::action`` marker is present but the block is malformed."""


@dataclass(frozen=True)
class Action:
    verb: str
    args: dict[str, str]
    body: str


def parse_action(text: str) -> Action | None:
    """Return the first well-formed Action in ``text``.

    Returns ``None`` if there is no ``::action`` marker at all (the model is
    just talking). Raises ``ProtocolError`` if a marker is present but the block
    cannot be parsed (e.g. missing ``::end`` or missing verb).
    """
    if "::action" not in text:
        return None

    match = _ACTION_RE.search(text)
    if match is None:
        raise ProtocolError("found '::action' but no closing '::end'")

    verb = match.group("header").strip()
    if not verb:
        raise ProtocolError("action is missing a verb")

    inner = match.group("inner")
    args, body = _split_args_and_body(inner)
    return Action(verb=verb, args=args, body=body)


def _split_args_and_body(inner: str) -> tuple[dict[str, str], str]:
    args: dict[str, str] = {}
    lines = inner.split("\n")
    body_lines: list[str] | None = None

    for i, line in enumerate(lines):
        if body_lines is not None:
            body_lines.append(line)
            continue
        if line.strip() == "---":
            body_lines = []
            continue
        if line.strip() == "":
            continue
        key, sep, value = line.partition(":")
        if sep == "":
            # A non key:value line before '---' starts the body implicitly.
            body_lines = lines[i:]
            break
        args[key.strip()] = value.strip()

    body = "\n".join(body_lines).strip("\n") if body_lines is not None else ""
    return args, body


def serialize_result(status: str, message: str) -> str:
    """Render an engine -> model result block."""
    if status not in ("ok", "error"):
        raise ValueError(f"status must be 'ok' or 'error', got {status!r}")
    return f"::result {status}\n{message}\n::end"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_protocol.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/protocol.py tests/test_protocol.py
git commit -m "feat: add action/result text protocol parser"
```

---

## Task 2: Sandbox path containment (`sandbox.py`)

**Files:**
- Create: `orchestrator/sandbox.py`
- Test: `tests/test_sandbox.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_sandbox.py`:
```python
import pytest

from orchestrator.sandbox import Sandbox, SandboxError


def test_resolve_simple_path(tmp_path):
    sandbox = Sandbox(tmp_path)
    resolved = sandbox.resolve("notes.md")
    assert resolved == (tmp_path / "notes.md").resolve()


def test_resolve_nested_path(tmp_path):
    sandbox = Sandbox(tmp_path)
    resolved = sandbox.resolve("sub/dir/file.txt")
    assert resolved == (tmp_path / "sub" / "dir" / "file.txt").resolve()


def test_resolve_dot_is_root(tmp_path):
    sandbox = Sandbox(tmp_path)
    assert sandbox.resolve(".") == tmp_path.resolve()


def test_escape_with_parent_raises(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.resolve("../secret.txt")


def test_absolute_path_outside_root_raises(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.resolve(str(tmp_path.parent / "elsewhere.txt"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.sandbox'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/sandbox.py`:
```python
"""The project folder is the trust boundary for first-party file tools."""
from __future__ import annotations

from pathlib import Path


class SandboxError(Exception):
    """Raised when a path would resolve outside the project folder."""


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` (relative to root, or absolute) and ensure it stays
        inside the project folder. Raises ``SandboxError`` otherwise."""
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxError(f"path escapes project folder: {path!r}")
        return candidate
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sandbox.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/sandbox.py tests/test_sandbox.py
git commit -m "feat: add project-folder sandbox path resolution"
```

---

## Task 3: First-party tools + registry (`tools.py`)

**Files:**
- Create: `orchestrator/tools.py`
- Test: `tests/test_tools.py`

**Verbs:** `read_file` (arg `path`), `write_file` (arg `path`, content from `body`), `list_dir` (arg `path`, default `.`), `run_command` (arg `cmd`). `run_command` runs `shell=True` with `cwd` = sandbox root and a timeout; it is cwd-scoped, not jailed (documented limitation).

- [ ] **Step 1: Write the failing tests**

`tests/test_tools.py`:
```python
import sys

import pytest

from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)


def test_write_then_read_file(registry, tmp_path):
    status, message = registry.execute(
        Action(verb="write_file", args={"path": "hello.txt"}, body="hi there")
    )
    assert status == "ok"
    assert (tmp_path / "hello.txt").read_text() == "hi there"

    status, message = registry.execute(
        Action(verb="read_file", args={"path": "hello.txt"}, body="")
    )
    assert status == "ok"
    assert "hi there" in message


def test_read_missing_file_is_error(registry):
    status, message = registry.execute(
        Action(verb="read_file", args={"path": "nope.txt"}, body="")
    )
    assert status == "error"
    assert "nope.txt" in message


def test_list_dir(registry, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    status, message = registry.execute(
        Action(verb="list_dir", args={"path": "."}, body="")
    )
    assert status == "ok"
    assert "a.txt" in message and "b.txt" in message


def test_run_command_captures_output(registry):
    status, message = registry.execute(
        Action(
            verb="run_command",
            args={"cmd": f'{sys.executable} -c "print(123)"'},
            body="",
        )
    )
    assert status == "ok"
    assert "123" in message


def test_unknown_verb_is_error(registry):
    status, message = registry.execute(
        Action(verb="fly_to_moon", args={}, body="")
    )
    assert status == "error"
    assert "fly_to_moon" in message


def test_sandbox_escape_is_error(registry):
    status, message = registry.execute(
        Action(verb="read_file", args={"path": "../escape.txt"}, body="")
    )
    assert status == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.tools'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/tools.py`:
```python
"""First-party, in-process tools. All paths are sandboxed; run_command is
cwd-scoped to the project folder (not jailed — a known limitation)."""
from __future__ import annotations

import subprocess
from typing import Callable

from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox, SandboxError

ToolHandler = Callable[[Action], str]


class ToolError(Exception):
    """Raised by a tool handler when an action cannot be completed."""


class ToolRegistry:
    def __init__(self, sandbox: Sandbox, command_timeout: float = 60.0) -> None:
        self.sandbox = sandbox
        self.command_timeout = command_timeout
        self._handlers: dict[str, ToolHandler] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_dir": self._list_dir,
            "run_command": self._run_command,
        }

    def execute(self, action: Action) -> tuple[str, str]:
        handler = self._handlers.get(action.verb)
        if handler is None:
            return "error", f"unknown verb: {action.verb}"
        try:
            return "ok", handler(action)
        except (ToolError, SandboxError) as exc:
            return "error", str(exc)

    def _require_arg(self, action: Action, key: str) -> str:
        value = action.args.get(key)
        if value is None:
            raise ToolError(f"{action.verb} requires arg '{key}'")
        return value

    def _read_file(self, action: Action) -> str:
        path = self.sandbox.resolve(self._require_arg(action, "path"))
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ToolError(f"no such file: {action.args['path']}")
        except OSError as exc:
            raise ToolError(f"read failed: {exc}")

    def _write_file(self, action: Action) -> str:
        rel = self._require_arg(action, "path")
        path = self.sandbox.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = action.body
        path.write_text(data, encoding="utf-8")
        return f"wrote {len(data.encode('utf-8'))} bytes to {rel}"

    def _list_dir(self, action: Action) -> str:
        rel = action.args.get("path", ".")
        path = self.sandbox.resolve(rel)
        if not path.is_dir():
            raise ToolError(f"not a directory: {rel}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return "\n".join(entries) if entries else "(empty)"

    def _run_command(self, action: Action) -> str:
        cmd = self._require_arg(action, "cmd")
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.sandbox.root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {self.command_timeout}s")
        output = (proc.stdout or "") + (proc.stderr or "")
        return f"exit={proc.returncode}\n{output}".rstrip()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/tools.py tests/test_tools.py
git commit -m "feat: add first-party file/command tools with registry"
```

---

## Task 4: LM Studio client (`llm_client.py`)

**Files:**
- Create: `orchestrator/llm_client.py`
- Test: `tests/test_llm_client.py`

Uses `httpx.MockTransport` so tests need no network and no extra deps.

- [ ] **Step 1: Write the failing tests**

`tests/test_llm_client.py`:
```python
import httpx
import pytest

from orchestrator.llm_client import LMStudioClient


def _client_with(handler) -> LMStudioClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return LMStudioClient(base_url="http://test/v1", timeout=5.0, http_client=http_client)


async def test_list_models_parses_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})

    client = _client_with(handler)
    assert await client.list_models() == ["model-a", "model-b"]


async def test_complete_returns_message_content():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = request.read().decode()
        assert "tools" not in body  # we never send the tools param
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hello from model"}}]},
        )

    client = _client_with(handler)
    out = await client.complete(
        model="model-a",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert out == "hello from model"


async def test_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = _client_with(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.complete(model="m", messages=[{"role": "user", "content": "x"}])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.llm_client'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/llm_client.py`:
```python
"""Async client for LM Studio's OpenAI-compatible API. Plain completions only —
we never send the ``tools`` parameter (see spec section 5)."""
from __future__ import annotations

import httpx


class LMStudioClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        timeout: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def list_models(self) -> list[str]:
        resp = await self._client.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    async def complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        resp = await self._client.post(
            f"{self.base_url}/chat/completions", json=payload
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm_client.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/llm_client.py tests/test_llm_client.py
git commit -m "feat: add async LM Studio completions client"
```

---

## Task 5: Single-agent loop (`agent.py`)

**Files:**
- Create: `orchestrator/agent.py`
- Test: `tests/test_agent.py`

**Loop:** build messages `[system, user(task), ...]`; call `client.complete`; append the assistant text; `parse_action` it.
- `None` → stop, `stopped_reason="no_action"`.
- malformed (`ProtocolError`) → append a corrective `::result error` user message showing the expected format and continue (counts as a step).
- verb `done` → stop, `stopped_reason="done"`.
- otherwise → `registry.execute`, append `serialize_result(...)` as a user message, continue.
- step cap reached → stop, `stopped_reason="max_steps"`.

The agent depends only on a client exposing `async complete(model, messages)` — tests pass a fake.

- [ ] **Step 1: Write the failing tests**

`tests/test_agent.py`:
```python
from orchestrator.agent import Agent, AgentResult
from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


class FakeClient:
    """Returns scripted completions in order, ignoring the prompt."""

    def __init__(self, scripted: list[str]) -> None:
        self._scripted = list(scripted)
        self.calls: list[list[dict]] = []

    async def complete(self, model, messages, temperature=0.7):
        self.calls.append(messages)
        return self._scripted.pop(0)


def _agent(tmp_path, scripted, max_steps=10):
    registry = ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)
    return Agent(
        client=FakeClient(scripted),
        registry=registry,
        model="test-model",
        system_prompt="You are a worker.",
        max_steps=max_steps,
    )


async def test_writes_file_then_done(tmp_path):
    scripted = [
        "I'll create the file.\n::action write_file\npath: out.txt\n---\nhi\n::end",
        "All set.\n::action done\n::end",
    ]
    agent = _agent(tmp_path, scripted)
    result = await agent.run("create out.txt with hi")
    assert isinstance(result, AgentResult)
    assert result.stopped_reason == "done"
    assert (tmp_path / "out.txt").read_text() == "hi"


async def test_result_is_fed_back_into_conversation(tmp_path):
    scripted = [
        "::action write_file\npath: a.txt\n---\nx\n::end",
        "::action done\n::end",
    ]
    agent = _agent(tmp_path, scripted)
    await agent.run("task")
    # Second model call must include the ::result ok from the first action.
    second_call_messages = agent.client.calls[1]
    assert any("::result ok" in m["content"] for m in second_call_messages)


async def test_no_action_stops(tmp_path):
    agent = _agent(tmp_path, ["I have nothing to do."])
    result = await agent.run("task")
    assert result.stopped_reason == "no_action"


async def test_malformed_action_gets_corrective_reprompt(tmp_path):
    scripted = [
        "::action write_file\npath: a.txt\n",  # missing ::end -> ProtocolError
        "::action done\n::end",
    ]
    agent = _agent(tmp_path, scripted)
    result = await agent.run("task")
    assert result.stopped_reason == "done"
    second_call_messages = agent.client.calls[1]
    assert any("::result error" in m["content"] for m in second_call_messages)


async def test_max_steps_stops(tmp_path):
    # Always emits a valid action, never 'done'.
    looping = "::action list_dir\npath: .\n::end"
    agent = _agent(tmp_path, [looping] * 5, max_steps=3)
    result = await agent.run("task")
    assert result.stopped_reason == "max_steps"
    assert len(agent.client.calls) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.agent'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/agent.py`:
```python
"""Single-agent text-protocol loop: emit -> parse one action -> execute ->
feed result back, until 'done', no action, or the step cap."""
from __future__ import annotations

from dataclasses import dataclass

from orchestrator.protocol import ProtocolError, parse_action, serialize_result
from orchestrator.tools import ToolRegistry

_FORMAT_REMINDER = (
    "Could not parse an action. Emit exactly one action block:\n"
    "::action <verb>\nkey: value\n---\noptional body\n::end\n"
    "When the task is finished, emit ::action done\\n::end."
)


@dataclass
class AgentResult:
    transcript: list[dict]
    stopped_reason: str


class Agent:
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

    async def run(self, task: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        reason = "max_steps"
        for _ in range(self.max_steps):
            reply = await self.client.complete(model=self.model, messages=messages)
            messages.append({"role": "assistant", "content": reply})

            try:
                action = parse_action(reply)
            except ProtocolError as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": serialize_result(
                            "error", f"{exc}\n\n{_FORMAT_REMINDER}"
                        ),
                    }
                )
                continue

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

        return AgentResult(transcript=messages, stopped_reason=reason)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_agent.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: PASS (27 passed).

- [ ] **Step 6: Commit**

```bash
git add orchestrator/agent.py tests/test_agent.py
git commit -m "feat: add single-agent text-protocol loop"
```

---

## Task 6: Config + real-LM-Studio smoke check

**Files:**
- Create: `orchestrator/config.py`
- Create: `orchestrator/smoke.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from orchestrator.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.lm_studio_url == "http://localhost:1234/v1"
    assert cfg.max_steps > 0
    assert cfg.command_timeout > 0
    assert cfg.request_timeout > 0


def test_override():
    cfg = Config(lm_studio_url="http://x/v1", max_steps=5)
    assert cfg.lm_studio_url == "http://x/v1"
    assert cfg.max_steps == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.config'`.

- [ ] **Step 3: Write `config.py`**

`orchestrator/config.py`:
```python
"""Central configuration. Phase 1 uses a subset; later phases extend it."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_mcp_json() -> str:
    return os.path.expanduser("~/.lmstudio/mcp.json")


@dataclass
class Config:
    lm_studio_url: str = "http://localhost:1234/v1"
    mcp_json_path: str = ""
    request_timeout: float = 120.0
    command_timeout: float = 60.0
    max_steps: int = 50

    def __post_init__(self) -> None:
        if not self.mcp_json_path:
            self.mcp_json_path = _default_mcp_json()

    def resolved_mcp_json(self) -> Path:
        return Path(self.mcp_json_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Write the smoke script**

`orchestrator/smoke.py`:
```python
"""Manual end-to-end smoke check against a running LM Studio.

Usage:
    python -m orchestrator.smoke <project_folder> [model_id]

Picks the first available model if none is given, then asks the agent to create
hello.txt containing "hello world" in the project folder.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from orchestrator.agent import Agent
from orchestrator.config import Config
from orchestrator.llm_client import LMStudioClient
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry

SYSTEM_PROMPT = (
    "You are an autonomous worker. Act using exactly one action block per reply:\n"
    "::action <verb>\nkey: value\n---\noptional body\n::end\n"
    "Verbs: read_file(path), write_file(path + body), list_dir(path), "
    "run_command(cmd). When the task is fully done, emit:\n::action done\n::end\n"
    "Think briefly in prose, then emit one action."
)


async def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m orchestrator.smoke <project_folder> [model_id]")
        return 2

    project = Path(sys.argv[1])
    project.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)

    try:
        model = sys.argv[2] if len(sys.argv) > 2 else (await client.list_models())[0]
        print(f"Using model: {model}")
        agent = Agent(
            client=client,
            registry=ToolRegistry(Sandbox(project), cfg.command_timeout),
            model=model,
            system_prompt=SYSTEM_PROMPT,
            max_steps=cfg.max_steps,
        )
        result = await agent.run(
            'Create a file named hello.txt containing exactly "hello world".'
        )
        print(f"Stopped: {result.stopped_reason}")
        target = project / "hello.txt"
        print(f"hello.txt exists: {target.exists()}")
        if target.exists():
            print(f"contents: {target.read_text()!r}")
    finally:
        await client.aclose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
```

- [ ] **Step 6: Run the smoke check (requires LM Studio running with a model loaded)**

Run: `python -m orchestrator.smoke ./scratch`
Expected: prints the chosen model, `Stopped: done` (or `no_action`), `hello.txt exists: True`, and contents containing `hello world`. If LM Studio is not running, it will raise a connection error — that is expected without the server and does not block the phase.

- [ ] **Step 7: Run full suite once more**

Run: `pytest -q`
Expected: PASS (29 passed).

- [ ] **Step 8: Commit**

```bash
git add orchestrator/config.py orchestrator/smoke.py tests/test_config.py
git commit -m "feat: add config and real-LM-Studio smoke script"
```

---

## Done criteria for Phase 1

- All unit tests pass (`pytest -q` green).
- `protocol.py`, `sandbox.py`, `tools.py`, `llm_client.py`, `agent.py`, `config.py` exist with the interfaces named above.
- The smoke script drives a real LM-Studio model through the text protocol to create a file in the sandbox.
- No native OpenAI tool-calling anywhere; all actions flow through the `::action`/`::result` protocol.

**Next phase (not in scope here):** `orchestrator.py` + `plan.py` + dual-agent turn-taking and `delegate`, then `mcp_host.py` for external tools (Exa), then the FastAPI + WebSocket UI with the kill switch.
