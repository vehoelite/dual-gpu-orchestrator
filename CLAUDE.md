# Dual-GPU Orchestrator — Project Guide

A local **autonomous dual-agent engine**. Two LM-Studio-served models cooperate to
drive a goal to completion with **no human input after kickoff** (the core
invariant). The program is a *client + orchestrator + tool host* — it does **not**
load models or manage GPUs; LM Studio already pins each model to a device.

> Note: `C:\Users\jacob\CLAUDE.md` (the "Walmart" guide) is a parent-directory file
> and is **unrelated** to this project. This file is the authoritative guide here.

## What it does

The user gives a goal. A three-tier pipeline runs it autonomously:

| Tier | Who | Job |
|------|-----|-----|
| **Planner** | frontier (Gemini) *or* the local dominant | Goal → ordered checklist |
| **Dominant** | local large model | Drive the checklist: delegate, review, mark done, complete. Never touches files. |
| **Worker** | local mid model | Execute one subtask at a time in a **fresh** context |

**Current rig (as of 2026-06-19) — same hardware, much larger models:**

| Role | Model id | GPU | Notes |
|------|----------|-----|-------|
| Dominant | `qwen3.5-122b-a10b-uncensored-hauhaucs-aggressive` | RTX 5070 (12 GB) / CUDA | 122B MoE, ~10B active (A10B); runs with **speculative decoding** using the 27B below as the draft model |
| Worker | `qwen3.6-27b-tq` | AMD RX 7600 (8 GB) / Vulkan-ROCm | also serves as the **draft model** for the dominant's speculative decoding |

The original verified rig (2026-06-15) was a 9B dominant + 4B worker; both roles
were upgraded in place. The engine is agnostic to model size — roles are pinned
by id substring in `cli.py` (e.g. `--dominant 122b --worker 27b`).

## Two key architectural decisions

1. **No native tool-calling.** Chosen because the original 4B worker was
   unreliable at OpenAI `tools` function-calling; the decision still stands with
   the larger models. All agent actions use a **structured text protocol** the
   engine parses. Agents emit one `::action <verb> … ::end` block per turn; the
   engine executes it and feeds back a `::result <ok|error> … ::end`. The parser
   (`protocol.py`) is deliberately forgiving (whitespace/fence tolerance, verb
   aliasing) — this is the project's highest-risk surface and its hardest-tested.
2. **First-party in-process tools, not MCP, for file/command work.** `tools.py`
   ships `read_file`/`write_file`/`list_dir`/`run_command`, all sandboxed to one
   project folder via `sandbox.py`. MCP is reserved for *external research only*
   (see Phase 3 below), and even that runs server-side in LM Studio — we write no
   MCP protocol code.

## Module map (`orchestrator/` package)

- `protocol.py` — parse `::action` blocks / serialize `::result`. Pure, heavily tested.
- `tools.py` + `sandbox.py` — first-party file/command tools; path containment to the project folder; `run_command` cwd = project folder.
- `llm_client.py` — async `httpx` wrapper over LM Studio OpenAI-compat `/v1` (plain completions, **no `tools` param**). Sends `LMSTUDIO_TOKEN` as Bearer when set.
- `agent.py` — the generic single-agent loop (emit → parse one action → execute → feed result). `terminal_verbs` is configurable (worker `{"done"}`, dominant `{"task_complete"}`). Loop is await-tolerant so registries can be sync or async. `run(task)` starts fresh; `resume(prior_transcript, followup)` continues an existing conversation (used by `retry`).
- `plan.py` — `Plan` / `Step` (pending|in_progress|done), `mark_done`/`revise`/`all_done`/`render`. Pure data.
- `planner.py` — `LocalPlanner` (prompts the local dominant) and `GeminiPlanner` (Google API; key from `GEMINI_API_KEY`, default model `gemini-3.5-flash`). The dashboard "premium planner" checkbox selects Gemini per-run (the plan is the highest-leverage artifact, so it's worth a frontier model while execution stays local); `server.build_planner` picks local vs gemini from the run params and falls back to local if the key is missing.
- `coordination.py` — `CoordinationRegistry`: the dominant's verbs (`set_plan`, `delegate`, `advance`, `retry`, `mark_done`, `revise_plan`, `task_complete`). `delegate`/`advance` build a **fresh** worker per NEW step — the dominant must paste literal content (URLs, values, file contents) into the body because a new worker shares no memory. `advance` = mark current done + delegate next in one turn (the eco happy path). `retry` re-runs the SAME step's worker via `Agent.resume` with its context intact (no re-paste; a one-line correction), so repeated failing retries trip the no-progress backstop. The in-progress step's live transcript is held in `_active_step`/`_active_transcript`, cleared when the step is marked done.
- `mcp_research.py` — `McpResearcher` + `mcp_integrations()`. Phase 3: a `research` verb backed by LM Studio's **native** `POST /api/v1/chat` with `integrations` from `mcp.json` (runs MCP servers server-side).
- `composite_registry.py` — routes `research` → `McpResearcher`, everything else → `ToolRegistry`.
- `orchestrator.py` — owns a run: plan → seed `Plan` → dominant loop + backstops → `RunResult`. Holds the `DOMINANT_PROMPT` / `WORKER_PROMPT` / `RESEARCH_HINT` / `DEBUG_HINT` system prompts and `worker_prompt_for(research, debug)` which composes the worker prompt. `DEBUG_HINT` (the dashboard "debug it" checkbox) makes the worker run + fix its own code IN-STEP (read-only/`--dry-run` only) — verification must be in the same subtask as creation, since a later step is a fresh worker with no memory of the files.
- `env.py` — tiny dependency-free `.env` loader (real env vars always win).
- `config.py` — `Config` dataclass (all defaults; backward-compatible across phases).
- `cli.py` — headless entry point (see below).
- `smoke.py` — live smoke helper.
- `events.py` — Phase 4 event layer: `make_event`, `preview`, `plan_event`, `NullSink`, `EventBus` (buffered fan-out to WebSocket subscribers; `reset()` drains subscriber queues). The agent loop, coordination, and orchestrator emit events into an optional sink.
- `run_manager.py` — `RunManager`: owns the single active run as an asyncio task wrapping a run-factory plus its `EventBus`. Hard-cancel kills the task and emits `run_aborted`; exceptions surface as an `error` event instead of crashing the server.
- `server.py` — FastAPI dashboard. REST (`models`, `run`, `stop`) configures/starts/stops the one run via `RunManager`; `/ws` replays the buffered events then streams live ones. `build_run_factory` is a module attribute so tests can monkeypatch it. Serves `static/`.
- `static/` — vanilla single-page dashboard (`index.html`, `app.js`, `style.css`): live log, plan view, kill switch. No build step.

## Running

```powershell
# tests (147 passing)
.venv\Scripts\python.exe -m pytest -q

# a headless run (LM Studio must be up with models loaded)
.venv\Scripts\python.exe -m orchestrator.cli "<goal>" .\scratch --dominant 9b --worker 4b

# the web dashboard (Phase 4) — then open http://127.0.0.1:8000
# use the module entry, not bare `uvicorn …:app`: main() calls load_dotenv()
# so LMSTUDIO_TOKEN reaches the /api/models + run endpoints.
.venv\Scripts\python.exe -m orchestrator.server
```

- `--dominant`/`--worker` are id substrings; without them, roles fall to LM Studio load order (arbitrary — pin them).
- Tests run on Windows via the venv interpreter; `asyncio_mode = "auto"`.

## Secrets & config

- `.env` (gitignored; `.env.example` committed) holds `GEMINI_API_KEY` and `LMSTUDIO_TOKEN`. `cli.py` calls `load_dotenv()` at startup. Real OS env vars take precedence.
- `LMSTUDIO_TOKEN` is required on **all** LM Studio endpoints once its API auth is on (sent as Bearer on the OpenAI-compat loop too, not just research).
- Research is enabled **only when** `LMSTUDIO_TOKEN` is set **and** `mcp.json` (`~/.lmstudio/mcp.json` by default) lists ≥1 server; otherwise the worker runs file/command only and everything else is unchanged.

## Status & phasing

- **Phase 1** (single-agent text-protocol loop), **Phase 2** (planner + dominant/worker orchestration + `delegate` + backstops), **Phase 3** (MCP research), **Phase 4** (web UI: event layer → `RunManager` → FastAPI `/ws` dashboard + kill switch) — all **implemented and merged**. 147 tests green.
- `pyproject.toml` now depends on `httpx`, `fastapi`, and `uvicorn[standard]`. Two entry points: the headless `cli.py` and the `server.py` dashboard (`uvicorn orchestrator.server:app`).
- Autonomy backstops still run underneath the UI: max-turns cap + no-progress detector + per-worker `max_steps`. The dashboard kill switch (`stop` → `RunManager.stop`) is now the live human touchpoint during a run.

## Docs

- Specs: `docs/superpowers/specs/` — top-level design (`2026-06-15-dual-gpu-orchestrator-design.md`) + per-phase designs.
- Plans: `docs/superpowers/plans/` — task-by-task implementation plans (phases 1–3).
- Reference: `docs/reference/lmstudio-mcp-via-api.md`, `docs/reference/lmstudio-rest-api.md` — the empirical LM Studio API findings that drove the Phase 3 native-endpoint decision.

## Conventions

- Python 3.11+, `httpx` for all HTTP (no provider SDKs), `pytest` + `pytest-asyncio`.
- Each module has one responsibility and is independently unit-tested; integration uses a **mock LM Studio** returning scripted `::action` text — the engine is built and verified without a real model. Keep that pattern.
- Never hard-code secrets; never commit `.env`.
