# Phase 3: MCP-backed Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the worker a `research` verb backed by LM Studio's native `/api/v1/chat` endpoint, which runs the user's `mcp.json` MCP servers (e.g. Exa) server-side — no MCP protocol code on our side.

**Architecture:** A `research` verb routes through a `CompositeRegistry` (first-party file/command tools stay on the synchronous `ToolRegistry`; `research` calls an async `McpResearcher` that POSTs to `/api/v1/chat` with `integrations` from `mcp.json` + a Bearer token). The worker stays 100% on the `::action` text protocol. Secrets load from a `.env` file (real env vars win).

**Tech Stack:** Python 3.11+, `httpx`, `pytest` + `pytest-asyncio`. No LM Studio SDK; no `python-dotenv`.

**Reference spec:** `docs/superpowers/specs/2026-06-14-phase3-mcp-research-design.md`
**API reference:** `docs/reference/lmstudio-mcp-via-api.md`

---

## File Structure

```
orchestrator/
  env.py                 # NEW: tiny dotenv loader
  mcp_research.py        # NEW: mcp_integrations() + McpResearcher (native /api/v1/chat)
  composite_registry.py  # NEW: routes research -> researcher, else -> ToolRegistry
  config.py              # MODIFY: lmstudio_native_url, research_model, research_timeout
  orchestrator.py        # MODIFY: add RESEARCH_HINT prompt snippet
  cli.py                 # MODIFY: load_dotenv(), build researcher+composite when enabled
.gitignore               # MODIFY: add .env
.env.example             # NEW: documents GEMINI_API_KEY / LMSTUDIO_TOKEN
tests/
  test_env.py              # NEW
  test_mcp_research.py     # NEW
  test_composite_registry.py # NEW
  test_config.py           # MODIFY (append 1 test)
  test_cli.py              # MODIFY (append 1 prompt test)
```

**Shared interfaces (define once, reuse — do not rename):**
- `load_dotenv(path=".env", environ=None) -> dict[str, str]` — `os.environ.setdefault` semantics (real env wins); `environ` injectable for tests.
- `mcp_integrations(mcp_json_path) -> list[str]` → `["mcp/<id>", ...]`; missing/malformed file → `[]`.
- `McpResearcher(base_url, token, model, integrations, http_client=None, timeout=180.0)` with `async research(query) -> str` and `async aclose()`.
- `CompositeRegistry(tool_registry, researcher)` with `async execute(action) -> tuple[str, str]`.
- `Config` gains `lmstudio_native_url="http://localhost:1234"`, `research_model=""`, `research_timeout=180.0`.

All pytest via `.venv\Scripts\python.exe -m pytest` (PowerShell, Windows; the tool cwd may be `C:\Users\jacob`, so use absolute paths or `git -C`). `asyncio_mode = "auto"`. Local git identity if needed: `git config user.email "vehoelite@gmail.com"; git config user.name "Jacob"`.

---

## Task 1: dotenv loader (`env.py`) + `.env` plumbing

**Files:**
- Create: `orchestrator/env.py`
- Create: `.env.example`
- Modify: `.gitignore`
- Test: `tests/test_env.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_env.py`:
```python
from orchestrator.env import load_dotenv


def test_parses_keys_comments_quotes_blanks(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        "# a comment\n"
        "\n"
        'GEMINI_API_KEY="abc123"\n'
        "LMSTUDIO_TOKEN = sk-lm-xyz \n"
        "NOT_A_PAIR\n",
        encoding="utf-8",
    )
    env = {}
    loaded = load_dotenv(f, environ=env)
    assert env["GEMINI_API_KEY"] == "abc123"
    assert env["LMSTUDIO_TOKEN"] == "sk-lm-xyz"
    assert "NOT_A_PAIR" not in env
    assert loaded == {"GEMINI_API_KEY": "abc123", "LMSTUDIO_TOKEN": "sk-lm-xyz"}


def test_does_not_override_existing(tmp_path):
    f = tmp_path / ".env"
    f.write_text("LMSTUDIO_TOKEN=from-file\n", encoding="utf-8")
    env = {"LMSTUDIO_TOKEN": "from-real-env"}
    load_dotenv(f, environ=env)
    assert env["LMSTUDIO_TOKEN"] == "from-real-env"  # real env wins


def test_missing_file_is_noop(tmp_path):
    assert load_dotenv(tmp_path / "nope.env", environ={}) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.env'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/env.py`:
```python
"""Tiny dependency-free .env loader so users can keep secrets in a file instead
of setting OS environment variables by hand. Real environment variables always
take precedence (we use setdefault)."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", environ: dict | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from ``path`` into ``environ`` (defaults to
    os.environ) without overriding existing values. Returns the pairs found in
    the file. Missing file is a no-op."""
    env = os.environ if environ is None else environ
    loaded: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        env.setdefault(key, value)
        loaded[key] = value
    return loaded
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_env.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Create `.env.example`**

`.env.example`:
```
# Copy this file to .env and fill in your values.
# .env is gitignored — never commit secrets.
GEMINI_API_KEY=
LMSTUDIO_TOKEN=
```

- [ ] **Step 6: Add `.env` to `.gitignore`**

Append a line so the file reads (add `.env` after the existing entries):
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
.env
```

- [ ] **Step 7: Commit**

```bash
git add orchestrator/env.py tests/test_env.py .env.example .gitignore
git commit -m "feat: add dependency-free .env loader and .env.example"
```

---

## Task 2: MCP researcher (`mcp_research.py`)

**Files:**
- Create: `orchestrator/mcp_research.py`
- Test: `tests/test_mcp_research.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_mcp_research.py`:
```python
import httpx
import pytest

from orchestrator.mcp_research import McpResearcher, mcp_integrations


def test_mcp_integrations_single(tmp_path):
    f = tmp_path / "mcp.json"
    f.write_text('{"mcpServers": {"exa": {"url": "https://x"}}}', encoding="utf-8")
    assert mcp_integrations(f) == ["mcp/exa"]


def test_mcp_integrations_multiple(tmp_path):
    f = tmp_path / "mcp.json"
    f.write_text(
        '{"mcpServers": {"exa": {"url": "x"}, "playwright": {"command": "y"}}}',
        encoding="utf-8",
    )
    assert sorted(mcp_integrations(f)) == ["mcp/exa", "mcp/playwright"]


def test_mcp_integrations_missing_or_bad(tmp_path):
    assert mcp_integrations(tmp_path / "nope.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert mcp_integrations(bad) == []


def _researcher(handler) -> McpResearcher:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return McpResearcher(
        base_url="http://localhost:1234", token="sk-lm-test",
        model="m", integrations=["mcp/exa"], http_client=client,
    )


async def test_research_request_shape_and_parse():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = __import__("json").loads(request.read())
        return httpx.Response(200, json={"output": [
            {"type": "reasoning", "content": "thinking"},
            {"type": "tool_call", "tool": "web_search_exa", "arguments": {}, "output": "..."},
            {"type": "message", "content": "the answer"},
        ]})

    r = _researcher(handler)
    out = await r.research("find news")
    assert captured["path"] == "/api/v1/chat"
    assert captured["auth"] == "Bearer sk-lm-test"
    assert captured["body"]["input"] == "find news"
    assert captured["body"]["integrations"] == ["mcp/exa"]
    assert "web_search_exa" in out
    assert "the answer" in out


async def test_research_handles_list_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"output": [
            {"type": "message", "content": [{"type": "output_text", "text": "part1 "},
                                            {"type": "output_text", "text": "part2"}]},
        ]})

    out = await _researcher(handler).research("q")
    assert "part1 part2" in out


async def test_research_http_error_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    with pytest.raises(httpx.HTTPStatusError):
        await _researcher(handler).research("q")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mcp_research.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.mcp_research'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/mcp_research.py`:
```python
"""MCP-backed research via LM Studio's native /api/v1/chat endpoint.

LM Studio runs the user's mcp.json servers (tool discovery + execution + loop)
server-side and returns the result. We write no MCP protocol code — we just call
the native endpoint with the configured server integrations. See
docs/reference/lmstudio-mcp-via-api.md."""
from __future__ import annotations

import json
from pathlib import Path

import httpx


def mcp_integrations(mcp_json_path) -> list[str]:
    """Return ['mcp/<id>', ...] for each server in mcp.json. Missing or
    unparseable file -> []."""
    p = Path(mcp_json_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    servers = data.get("mcpServers", {})
    return [f"mcp/{name}" for name in servers]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        return "".join(parts)
    return "" if content is None else str(content)


def _extract_answer(output: list) -> str:
    """Take the last `type == "message"` entry as the answer; prefix a note of
    which tools ran."""
    if not isinstance(output, list):
        return str(output)[:2000]
    messages = [o for o in output if isinstance(o, dict) and o.get("type") == "message"]
    text = _content_text(messages[-1].get("content")) if messages else ""
    tools = [
        o.get("tool")
        for o in output
        if isinstance(o, dict) and o.get("type") == "tool_call" and o.get("tool")
    ]
    note = f"[used: {', '.join(tools)}]\n" if tools else ""
    return (note + text).strip() or json.dumps(output)[:2000]


class McpResearcher:
    def __init__(
        self,
        base_url: str,
        token: str,
        model: str,
        integrations: list[str],
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.model = model
        self.integrations = integrations
        self._client = http_client or httpx.AsyncClient(timeout=timeout)

    async def research(self, query: str) -> str:
        payload = {
            "model": self.model,
            "input": query,
            "integrations": self.integrations,
        }
        resp = await self._client.post(
            f"{self.base_url}/api/v1/chat",
            json=payload,
            headers={"Authorization": f"Bearer {self.token}"},
        )
        resp.raise_for_status()
        return _extract_answer(resp.json().get("output", []))

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_mcp_research.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/mcp_research.py tests/test_mcp_research.py
git commit -m "feat: add McpResearcher over LM Studio native /api/v1/chat"
```

---

## Task 3: Composite registry (`composite_registry.py`)

**Files:**
- Create: `orchestrator/composite_registry.py`
- Test: `tests/test_composite_registry.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_composite_registry.py`:
```python
from orchestrator.composite_registry import CompositeRegistry
from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


class FakeResearcher:
    def __init__(self, answer="research answer", boom=False):
        self.answer = answer
        self.boom = boom
        self.received = None

    async def research(self, query):
        self.received = query
        if self.boom:
            raise RuntimeError("network down")
        return self.answer


def _composite(tmp_path, researcher):
    return CompositeRegistry(
        tool_registry=ToolRegistry(Sandbox(tmp_path), command_timeout=10.0),
        researcher=researcher,
    )


async def test_research_routes_to_researcher(tmp_path):
    r = FakeResearcher(answer="found it")
    comp = _composite(tmp_path, r)
    status, msg = await comp.execute(Action("research", {"query": "news?"}, ""))
    assert status == "ok"
    assert msg == "found it"
    assert r.received == "news?"


async def test_research_uses_body_when_no_query_arg(tmp_path):
    r = FakeResearcher()
    comp = _composite(tmp_path, r)
    await comp.execute(Action("research", {}, "  body query  "))
    assert r.received == "body query"


async def test_research_missing_query_is_error(tmp_path):
    comp = _composite(tmp_path, FakeResearcher())
    status, msg = await comp.execute(Action("research", {}, "   "))
    assert status == "error"


async def test_research_exception_is_error(tmp_path):
    comp = _composite(tmp_path, FakeResearcher(boom=True))
    status, msg = await comp.execute(Action("research", {"query": "x"}, ""))
    assert status == "error"
    assert "network down" in msg


async def test_file_verb_routes_to_tool_registry(tmp_path):
    comp = _composite(tmp_path, FakeResearcher())
    status, msg = await comp.execute(
        Action("write_file", {"path": "a.txt"}, "hi")
    )
    assert status == "ok"
    assert (tmp_path / "a.txt").read_text() == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_composite_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.composite_registry'`.

- [ ] **Step 3: Write the implementation**

`orchestrator/composite_registry.py`:
```python
"""Routes the `research` verb to an McpResearcher (async, network) and every
other verb to the first-party ToolRegistry (sync). The Phase 2 agent loop is
await-tolerant, so this async execute() drops in for the worker."""
from __future__ import annotations

from orchestrator.protocol import Action


class CompositeRegistry:
    def __init__(self, tool_registry, researcher) -> None:
        self.tool_registry = tool_registry
        self.researcher = researcher

    async def execute(self, action: Action) -> tuple[str, str]:
        if action.verb == "research":
            query = action.args.get("query") or action.body.strip()
            if not query:
                return "error", "research needs a 'query' arg or a body"
            try:
                answer = await self.researcher.research(query)
            except Exception as exc:  # surface any failure to the agent, don't crash
                return "error", f"research failed: {exc}"
            return "ok", answer
        return self.tool_registry.execute(action)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_composite_registry.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/composite_registry.py tests/test_composite_registry.py
git commit -m "feat: add CompositeRegistry routing research vs first-party tools"
```

---

## Task 4: Config additions (`config.py`)

**Files:**
- Modify: `orchestrator/config.py`
- Test: `tests/test_config.py` (append 1 test)

- [ ] **Step 1: Append the failing test to `tests/test_config.py`**

```python
def test_phase3_defaults():
    cfg = Config()
    assert cfg.lmstudio_native_url == "http://localhost:1234"
    assert cfg.research_model == ""
    assert cfg.research_timeout > 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py::test_phase3_defaults -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'lmstudio_native_url'`.

- [ ] **Step 3: Add the fields**

In `orchestrator/config.py`, the `Config` dataclass currently ends its field block with the Phase 2 fields:
```python
    dominant_model: str = ""
    worker_model: str = ""
```
Add the Phase 3 fields immediately after `worker_model` (before `__post_init__`):
```python
    dominant_model: str = ""
    worker_model: str = ""
    # Phase 3: MCP research
    lmstudio_native_url: str = "http://localhost:1234"
    research_model: str = ""
    research_timeout: float = 180.0
```

- [ ] **Step 4: Run the config tests to verify they pass**

Run: `.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: PASS (all config tests, including the new one).

- [ ] **Step 5: Commit**

```bash
git add orchestrator/config.py tests/test_config.py
git commit -m "feat: add Phase 3 config fields (native url, research model/timeout)"
```

---

## Task 5: Wire research into the CLI (`cli.py`, `orchestrator.py`)

**Files:**
- Modify: `orchestrator/orchestrator.py` (add `RESEARCH_HINT`)
- Modify: `orchestrator/cli.py`
- Test: `tests/test_cli.py` (append 1 test)

- [ ] **Step 1: Append the failing test to `tests/test_cli.py`**

```python
def test_research_hint_has_concrete_example():
    from orchestrator.orchestrator import RESEARCH_HINT
    assert "::action research" in RESEARCH_HINT
    assert "query:" in RESEARCH_HINT
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::test_research_hint_has_concrete_example -v`
Expected: FAIL with `ImportError: cannot import name 'RESEARCH_HINT'`.

- [ ] **Step 3: Add `RESEARCH_HINT` to `orchestrator/orchestrator.py`**

Immediately after the `WORKER_PROMPT = (...)` definition, add:
```python
RESEARCH_HINT = (
    "\n\nYou also have web/tool research via MCP. Use it for facts you don't know:\n"
    "::action research\n"
    "query: latest NVIDIA data-center news with sources\n"
    "::end\n"
    "The result is a synthesized answer from external tools."
)
```

- [ ] **Step 4: Run the prompt test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cli.py::test_research_hint_has_concrete_example -v`
Expected: PASS.

- [ ] **Step 5: Wire `cli.py`** — replace the current `main()` body's setup so it loads `.env`, builds the researcher when enabled, and composes the worker registry.

In `orchestrator/cli.py`, update the imports block to add the new modules:
```python
from orchestrator.agent import Agent
from orchestrator.composite_registry import CompositeRegistry
from orchestrator.config import Config
from orchestrator.env import load_dotenv
from orchestrator.llm_client import LMStudioClient
from orchestrator.mcp_research import McpResearcher, mcp_integrations
from orchestrator.orchestrator import RESEARCH_HINT, Orchestrator, WORKER_PROMPT
from orchestrator.planner import GeminiPlanner, LocalPlanner
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry
```
(Keep `argparse`, `asyncio`, `os`, `Path` imports as they are.)

Then replace the `main()` function body. The current body opens with:
```python
async def main() -> int:
    args = _parse_args()
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)
```
Insert `load_dotenv()` as the very first line of `main()` (before `args = _parse_args()`):
```python
async def main() -> int:
    load_dotenv()  # populate os.environ from .env (real env wins)
    args = _parse_args()
    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)
```

Then, after the model-selection block (which sets `cfg.dominant_model` / `cfg.worker_model` and prints them), and BEFORE the existing `def worker_factory()`, add the research setup:
```python
        # Phase 3: enable MCP research when a token + mcp.json servers are present.
        token = os.environ.get("LMSTUDIO_TOKEN", "")
        integrations = mcp_integrations(cfg.resolved_mcp_json())
        researcher = None
        worker_prompt = WORKER_PROMPT
        if token and integrations:
            researcher = McpResearcher(
                base_url=cfg.lmstudio_native_url,
                token=token,
                model=cfg.research_model or cfg.worker_model,
                integrations=integrations,
                timeout=cfg.research_timeout,
            )
            worker_prompt = WORKER_PROMPT + RESEARCH_HINT
            print(f"research enabled via {integrations}")
        else:
            print("research disabled (set LMSTUDIO_TOKEN and configure mcp.json to enable)")
```

Replace the existing `worker_factory` definition with one that uses `worker_prompt` and wraps in a `CompositeRegistry` when research is enabled:
```python
        def worker_factory() -> Agent:
            tool_registry = ToolRegistry(Sandbox(project), cfg.command_timeout)
            registry = (
                CompositeRegistry(tool_registry, researcher)
                if researcher is not None
                else tool_registry
            )
            return Agent(
                client=client,
                registry=registry,
                model=cfg.worker_model,
                system_prompt=worker_prompt,
                max_steps=cfg.max_steps,
            )
```

Finally, ensure the researcher's HTTP client is closed. The existing `finally` block closes the LM Studio client:
```python
    finally:
        await client.aclose()
    return 0
```
Replace it with:
```python
    finally:
        await client.aclose()
        if researcher is not None:
            await researcher.aclose()
    return 0
```
Because `researcher` is referenced in `finally`, initialize it to `None` BEFORE the `try:` (right after `client = LMStudioClient(...)`):
```python
    cfg = Config()
    client = LMStudioClient(base_url=cfg.lm_studio_url, timeout=cfg.request_timeout)
    researcher = None
    try:
```
(The later assignment inside the `try` reuses this name.)

- [ ] **Step 6: Verify the CLI still shows usage without a server**

Run: `.venv\Scripts\python.exe -m orchestrator.cli`
Expected: argparse prints `usage: ...` and exits with code 2 (no server, no `.env` needed — `load_dotenv()` is a no-op when there's no `.env`).

- [ ] **Step 7: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: PASS, 0 failures.

- [ ] **Step 8: Commit**

```bash
git add orchestrator/cli.py orchestrator/orchestrator.py tests/test_cli.py
git commit -m "feat: wire MCP research into CLI with .env and conditional enablement"
```

---

## Done criteria for Phase 3

- Full suite green (0 failures).
- New modules exist with the named interfaces; `Config` extended backward-compatibly.
- Worker gains a `research` verb ONLY when `LMSTUDIO_TOKEN` + `mcp.json` servers are present; otherwise the engine runs exactly as in Phase 2.
- `research` routes to LM Studio's native `/api/v1/chat` with `integrations`; the worker stays on the `::action` text protocol.
- Secrets load from `.env` (real env wins); `.env` is gitignored; `.env.example` is committed; no secret is hard-coded.

**Manual live smoke (optional, outside the unit suite):** put `LMSTUDIO_TOKEN=<token>` in `.env`, ensure LM Studio 0.4.0+ has "Allow calling servers from mcp.json" enabled with the Exa server, then:
```
.venv\Scripts\python.exe -m orchestrator.cli "find this week's top NVIDIA headline with a source and write it to news.txt" .\scratch --dominant 9b --worker 4b
```
Expect the worker to emit `::action research`, get live results, and write `news.txt`.

**Next phase (not in scope):** Phase 4 — FastAPI + WebSocket UI + interactive kill switch.
