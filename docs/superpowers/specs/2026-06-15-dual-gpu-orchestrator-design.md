# Dual-GPU Autonomous Dual-Agent Engine — Design

**Date:** 2026-06-15
**Status:** Approved (design phase)

## 1. Summary

A local autonomous **dual-agent engine** with a web UI. Two models — already
loaded and GPU-pinned by **LM Studio** — are assigned a "dominant" and a
"worker" role. The user gives the dominant a goal; the dominant writes a
plan/checklist and drives the worker through conversational turns until every
checklist item is complete. Both agents can use MCP tools to read/write files
and run commands, all confined to a single project folder.

Verified environment (2026-06-15):

| Role | Model (LM Studio id) | GPU | Backend |
|------|----------------------|-----|---------|
| Dominant | `omnicoder-qwen3.5-9b-claude-4.6-opus-uncensored-v2` (9B) | NVIDIA RTX 5070 (12 GB) | CUDA (llama.cpp) |
| Worker | `qwen3.5-4b-claude-4.6-opus-reasoning-distill-heretic-v3-i1` (4B) | AMD RX 7600 (8 GB) | Vulkan/ROCm (llama.cpp) |

**Key decision:** This program does **not** load models or manage GPUs. LM Studio
already solves model loading and device placement across the heterogeneous
NVIDIA + AMD setup (the genuinely hard part). Our program is purely a
**client + orchestrator + MCP host** talking to LM Studio's OpenAI-compatible API
(`http://localhost:1234/v1`).

## 2. Goals / Non-Goals

**Goals**
- Assign dominant/worker roles to two LM-Studio-served models from a web UI.
- Autonomous loop: dominant plans, both agents trade turns, both use MCP tools,
  dominant sees the plan through to completion. No human in the loop after the
  task is given.
- Reuse the user's existing LM Studio `mcp.json` for MCP server definitions.
- Confine all filesystem/command activity to one project folder (sandbox).
- Live web dashboard: configure a run, watch the conversation + tool calls +
  plan checklist, kill switch.

**Non-Goals**
- No raw PyTorch / transformers model loading; no GPU pinning logic.
- No multi-user / remote / auth concerns (local single-user tool).
- No approval gates for individual actions — anything is allowed *inside* the
  project folder.

## 3. Architecture

Each component has one purpose, a defined interface, and is independently
testable.

- **`llm_client.py`** — async wrapper over LM Studio's `/v1/models` and
  `/v1/chat/completions` (with `tools`). Knows nothing about agents or MCP.
- **`mcp_host.py`** — reads the same `mcp.json` LM Studio uses; launches each MCP
  server **scoped to the project folder** (filesystem server rooted there, shell
  servers with cwd set there); aggregates tool schemas into OpenAI tool format;
  executes tool calls. Uses the Python `mcp` SDK over stdio transport.
- **`agent.py`** — one agent = (model id, role, system prompt, tool set). Runs a
  single-turn tool-calling loop: emit → execute any `tool_calls` via `mcp_host`
  → feed results back → repeat until the model returns control (plain message or
  a control tool).
- **`plan.py`** — parses and stores the dominant's checklist; tracks each step's
  status (`pending` / `in_progress` / `done`).
- **`orchestrator.py`** — owns a run: alternates turns dominant↔worker, routes
  `delegate_to_worker`, applies plan updates, enforces max-turns + no-progress
  detection + kill switch, emits events to subscribers.
- **`server.py`** — FastAPI app: REST (`GET /models`, `POST /run`, `POST /stop`)
  + WebSocket (`/ws` live event stream).
- **`static/`** — single-page UI: role/model dropdowns (auto-filled from LM
  Studio), project-folder + task inputs, MCP server toggles, live conversation +
  tool-call log, plan checklist view, kill switch.
- **`config.py` / `sandbox.py`** — settings (LM Studio base URL, mcp.json path,
  limits), project-folder resolution and path-containment checks.

## 4. Control Flow

1. UI loads → `GET /models` lists LM Studio's available models → user picks
   dominant + worker, sets project folder + task, toggles MCP servers →
   `POST /run`.
2. Orchestrator gives the **dominant** the task; dominant calls
   `set_plan(checklist)`.
3. Turn loop: dominant acts (may call MCP tools and/or
   `delegate_to_worker(subtask, context)`). On delegation, the **worker** runs
   its own tool-calling loop on the subtask and returns a result; the dominant
   reviews it and calls `mark_done(step)` or `revise_plan(...)`.
4. Repeat until every checklist item is `done` → dominant calls `task_complete`
   → run ends.
5. Every step streams to the UI over WebSocket.

**Termination:** primary = checklist complete (`task_complete`). Backstops =
max-turns cap, no-progress detection (N turns with no plan change), and the
manual kill switch.

## 5. Tooling Model

Coordination uses the **native OpenAI tool-calling loop** (LM Studio `tools`
API). We inject "control" tools alongside the MCP tools:

- **Dominant control tools:** `set_plan`, `mark_done`, `revise_plan`,
  `delegate_to_worker`, `task_complete`.
- **Both agents:** all MCP tools from `mcp.json` (filesystem, shell, etc.),
  sandboxed to the project folder.

Delegation and plan tracking are therefore ordinary tool calls, keeping the
engine uniform.

## 6. Sandbox

The project folder is the trust boundary. Within it, agents may do anything;
nothing escapes it.

- Filesystem MCP server is relaunched **rooted at the project folder** rather
  than whatever path LM Studio's config uses.
- Shell/command MCP servers run with **cwd = project folder**.
- `sandbox.py` validates any path arguments the engine itself handles for
  containment; servers themselves enforce their own roots.
- If the LM Studio `mcp.json` path is non-standard, it is configurable in
  `config.py`.

## 7. Error Handling

| Scenario | Handling |
|----------|----------|
| LM Studio unreachable / model not loaded | Caught on `GET /models` or run start; surfaced in UI before the loop begins. |
| Malformed tool-call JSON from a model | Corrective retry with an explanatory message; capped retry count, then skip/abort the turn gracefully. |
| MCP server fails to launch | Reported as an event; run continues with the remaining tools. |
| Runaway / no progress | Max-turns cap + no-progress detector stop the run. |
| Kill switch pressed | Cancel in-flight requests, stop MCP servers, mark run aborted. |

## 8. Testing

- **Unit:** `plan.py` (parse/track transitions); `mcp_host` schema aggregation +
  call routing (mocked MCP servers); `llm_client` (mocked HTTP).
- **Deterministic integration:** a **mock LM Studio** server returning scripted
  `tool_calls` drives `orchestrator.py` through full dominant/worker turn
  sequences without a real model.
- **Smoke:** one test against real LM Studio with a trivial task (e.g. "create
  hello.txt in the project folder").

## 9. Stack & Phasing

**Stack:** Python 3.11+, `httpx` (async), `fastapi` + `uvicorn`, `mcp`
(Python SDK), vanilla-JS single-page frontend.

**Phases**
1. `llm_client` + `agent` + `mcp_host` + a single-agent tool-calling loop working
   end-to-end against LM Studio.
2. `orchestrator` + `plan` + dual-agent turn-taking and delegation.
3. Web UI + WebSocket streaming + kill switch.

## 10. Risks

- **4B worker tool-calling reliability** is the main unknown. Mitigations:
  corrective-retry backstop, and the mock-LM-Studio tests let us build/verify the
  engine independent of model quirks. If the 4B proves too weak, fall back to a
  more structured delegation protocol (kept out of scope for now).
