# Dual-GPU Autonomous Dual-Agent Engine — Design

**Date:** 2026-06-15 (rev. 2026-06-14 — tooling model)
**Status:** Work in progress (design approved; text-protocol revision)

## 1. Summary

A local autonomous **dual-agent engine** with a web UI. Two models — already
loaded and GPU-pinned by **LM Studio** — are assigned a "dominant" and a
"worker" role. **Conceptually the dominant plays the seat a human normally
occupies in today's agent workflows** (it sets the goal-into-plan, delegates,
reviews, and decides when the work is done), while the worker is the AI agent
that executes tasks. The product's purpose is to **automate the human seat** so
the loop runs to completion with nobody in the chair.

The user gives the dominant a goal; the dominant writes a plan/checklist and
drives the worker through conversational turns until every checklist item is
complete. Both agents act on the project — read/write files, run commands — via
**first-party tools** the engine ships with, all confined to a single project
folder.

**Core invariant: full autonomy after kickoff.** Once a run starts, the system
requires **no human input**. There are no per-action approval gates; the only
human touchpoint is an optional kill switch.

Verified environment (2026-06-15):

| Role | Model (LM Studio id) | GPU | Backend |
|------|----------------------|-----|---------|
| Dominant | `omnicoder-qwen3.5-9b-claude-4.6-opus-uncensored-v2` (9B) | NVIDIA RTX 5070 (12 GB) | CUDA (llama.cpp) |
| Worker | `qwen3.5-4b-claude-4.6-opus-reasoning-distill-heretic-v3-i1` (4B) | AMD RX 7600 (8 GB) | Vulkan/ROCm (llama.cpp) |

**Key decision 1 — no model/GPU management.** This program does **not** load
models or manage GPUs. LM Studio already solves model loading and device
placement across the heterogeneous NVIDIA + AMD setup (the genuinely hard part).
Our program is purely a **client + orchestrator + tool host** talking to LM
Studio's OpenAI-compatible API (`http://localhost:1234/v1`).

**Key decision 2 — first-party tools, not native tool-calling.** The engine
ships its own in-process filesystem/command tools rather than depending on
whatever MCP servers a user has configured. `mcp.json` is reserved for genuinely
external tools (e.g. Exa web search). Critically, agents do **not** use the
OpenAI tool-calling (`tools`) API at all — some target models (especially the
4B worker) are unreliable at native function-calling. Instead, all actions are
expressed through a **structured text protocol** the engine parses (see §5).

## 2. Goals / Non-Goals

**Goals**
- Assign dominant/worker roles to two LM-Studio-served models from a web UI.
- Autonomous loop: dominant plans, both agents trade turns, both act via tools,
  dominant sees the plan through to completion. **No human in the loop after the
  task is given** (core invariant).
- Ship first-party in-process filesystem/command tools; reserve `mcp.json` for
  external tools (e.g. Exa).
- Drive all agent actions through a structured text protocol (no native
  tool-calling), tolerant of weak models.
- Confine all filesystem/command activity to one project folder (sandbox).
- Live web dashboard: configure a run, watch the conversation + actions + plan
  checklist, kill switch.

**Non-Goals**
- No raw PyTorch / transformers model loading; no GPU pinning logic.
- No multi-user / remote / auth concerns (local single-user tool).
- No native OpenAI tool-calling (`tools` API) — replaced by the text protocol.
- No approval gates for individual actions — anything is allowed *inside* the
  project folder.

## 3. Architecture

Each component has one purpose, a defined interface, and is independently
testable.

- **`llm_client.py`** — async wrapper over LM Studio's `/v1/models` and
  `/v1/chat/completions`. **Plain completions, no `tools` param.** Knows nothing
  about agents, protocol, or tools.
- **`protocol.py`** — the heart of the system. Serializes `::result` blocks and
  parses the `::action` blocks models emit (see §5). Deliberately forgiving:
  whitespace/fence tolerance, verb aliasing, clear parse errors. Pure functions,
  unit-tested hard.
- **`tools.py`** — first-party in-process tool implementations
  (`read_file`, `write_file`, `list_dir`, `run_command`) plus a registry mapping
  action verbs → callables. Each tool enforces the sandbox via `sandbox.py`.
- **`mcp_host.py`** — thin client for **external** MCP servers declared in
  `mcp.json` (e.g. Exa). Connects over the Python `mcp` SDK and registers each
  external tool into the same registry, surfaced through the same text protocol
  so the model sees one uniform action mechanism.
- **`agent.py`** — one agent = (model id, role, system prompt, available verbs).
  Runs the loop: emit text → parse **one** `::action` → execute via the registry
  → append a `::result` → repeat until a control action returns the turn.
- **`plan.py`** — parses and stores the dominant's checklist; tracks each step's
  status (`pending` / `in_progress` / `done`).
- **`orchestrator.py`** — owns a run: alternates turns dominant↔worker, routes
  `delegate`, applies plan updates, enforces max-turns + no-progress detection +
  kill switch, emits events to subscribers.
- **`server.py`** — FastAPI app: REST (`GET /models`, `POST /run`, `POST /stop`)
  + WebSocket (`/ws` live event stream).
- **`static/`** — single-page UI: role/model dropdowns (auto-filled from LM
  Studio), project-folder + task inputs, external-tool (Exa) toggle, live
  conversation + action log, plan checklist view, kill switch.
- **`config.py` / `sandbox.py`** — settings (LM Studio base URL, mcp.json path,
  limits), project-folder resolution and path-containment checks.

## 4. Control Flow

1. UI loads → `GET /models` lists LM Studio's available models → user picks
   dominant + worker, sets project folder + task, toggles external tools →
   `POST /run`. **This is the last human input of the run.**
2. Orchestrator gives the **dominant** the task; dominant emits a `set_plan`
   action with the checklist.
3. Turn loop: dominant acts (file/command actions and/or a `delegate` action
   with subtask + context). On delegation, the **worker** runs its own
   one-action-per-turn loop on the subtask and returns a result; the dominant
   reviews it and emits `mark_done` or `revise_plan`.
4. Repeat until every checklist item is `done` → dominant emits `task_complete`
   → run ends.
5. Every action and result streams to the UI over WebSocket.

**Termination:** primary = checklist complete (`task_complete`). Backstops =
max-turns cap, no-progress detection (N turns with no plan change), and the
manual kill switch.

## 5. Tooling Model — Structured Text Protocol

No native OpenAI tool-calling. Agents act by emitting a structured text block in
their normal output; the engine parses it, executes the action, and feeds a
`::result` back on the next turn. One uniform mechanism for first-party tools,
coordination, and external (MCP) tools alike.

**Action format:**

```
::action <verb>
<key>: <value>
---
<optional freeform body — e.g. file contents>
::end
```

**Result format (engine → model):**

```
::result <ok|error>
<message / captured output>
::end
```

**One action per turn.** The model reasons in prose, then emits exactly one
`::action` and yields. The engine executes it and returns one `::result`, then
the model continues. This keeps parsing unambiguous and prevents weak models
from emitting multiple half-formed actions. Cost: more turns (acceptable).

**Verb families (all via the same parser + registry):**

- **First-party filesystem/command (in-process):** `read_file`, `write_file`,
  `list_dir`, `run_command` — executed directly by `tools.py`, sandboxed to the
  project folder.
- **Coordination:** `set_plan`, `mark_done`, `revise_plan`, `delegate`
  (dominant→worker), `task_complete`. These mutate plan/run state rather than
  the filesystem.
- **External (MCP):** e.g. `web_search` (Exa), proxied by `mcp_host.py` but
  presented to the model identically.

Available verbs are role-scoped: the worker gets file/command + external verbs;
the dominant additionally gets the coordination verbs.

## 6. Sandbox

The project folder is the trust boundary. Within it, agents may do anything;
nothing escapes it.

- First-party tools are in-process, so containment is enforced directly:
  `sandbox.py` resolves every path argument and rejects anything outside the
  project folder before `tools.py` touches the filesystem.
- `run_command` executes with **cwd = project folder**.
- External MCP servers (e.g. Exa) are network tools, not filesystem tools, so
  they pose no path-escape risk; if a future external server did touch the
  filesystem it would be launched scoped to the project folder.
- The LM Studio `mcp.json` path is configurable in `config.py`.

## 7. Error Handling

| Scenario | Handling |
|----------|----------|
| LM Studio unreachable / model not loaded | Caught on `GET /models` or run start; surfaced in UI before the loop begins. |
| Unparseable / malformed `::action` block | One corrective re-prompt showing the exact expected format; capped retries, then skip/abort the turn gracefully. |
| Unknown verb or bad arguments | `::result error` with a specific message; the model retries on its next turn. |
| External MCP server fails to launch | Reported as an event; run continues with first-party tools only. |
| Runaway / no progress | Max-turns cap + no-progress detector stop the run automatically (no human needed). |
| Kill switch pressed | Cancel in-flight requests, stop external servers, mark run aborted. |

## 8. Testing

- **Unit:** `protocol.py` (parse/serialize, incl. malformed/edge-case blocks —
  the most important suite); `tools.py` + `sandbox.py` (path containment, command
  cwd); `plan.py` (parse/track transitions); `mcp_host` call routing (mocked
  external server); `llm_client` (mocked HTTP).
- **Deterministic integration:** a **mock LM Studio** server returning scripted
  text completions (containing `::action` blocks) drives `orchestrator.py`
  through full dominant/worker turn sequences without a real model.
- **Smoke:** one test against real LM Studio with a trivial task (e.g. "create
  hello.txt in the project folder").

## 9. Stack & Phasing

**Stack:** Python 3.11+, `httpx` (async), `fastapi` + `uvicorn`, `mcp`
(Python SDK, external servers only), vanilla-JS single-page frontend.

**Phases**
1. `protocol` + `tools` + `sandbox` + `llm_client` + `agent` — a single-agent
   text-protocol loop working end-to-end against LM Studio (create/read files,
   run commands in the sandbox).
2. `orchestrator` + `plan` + dual-agent turn-taking and `delegate`; external
   tools via `mcp_host`.
3. Web UI + WebSocket streaming + kill switch.

## 10. Risks

- **Protocol-following reliability** is the main unknown — can the models
  (especially the 4B worker) consistently emit well-formed `::action` blocks?
  Mitigations: a very forgiving parser (whitespace/fence tolerance, verb
  aliasing), one corrective re-prompt showing the exact format on a parse miss,
  one-action-per-turn to minimize what the model must get right at once, and the
  mock-LM-Studio tests so the engine is built/verified independent of model
  quirks. The text protocol is itself the chosen mitigation for the original
  "weak at native tool-calling" risk.
