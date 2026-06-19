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
- `agent.py` — the generic single-agent loop (emit → parse one action → execute → feed result). `terminal_verbs` is configurable (worker `{"done"}`, dominant `{"task_complete"}`). Loop is await-tolerant so registries can be sync or async.
- `plan.py` — `Plan` / `Step` (pending|in_progress|done), `mark_done`/`revise`/`all_done`/`render`. Pure data.
- `planner.py` — `LocalPlanner` (prompts the 9B) and `GeminiPlanner` (Google API; key from `GEMINI_API_KEY`).
- `coordination.py` — `CoordinationRegistry`: the dominant's verbs (`set_plan`, `delegate`, `mark_done`, `revise_plan`, `task_complete`). `delegate` builds a **fresh** worker per subtask — the dominant must paste literal content (URLs, values, file contents) into each delegate body because workers share no memory.
- `mcp_research.py` — `McpResearcher` + `mcp_integrations()`. Phase 3: a `research` verb backed by LM Studio's **native** `POST /api/v1/chat` with `integrations` from `mcp.json` (runs MCP servers server-side).
- `composite_registry.py` — routes `research` → `McpResearcher`, everything else → `ToolRegistry`.
- `orchestrator.py` — owns a run: plan → seed `Plan` → dominant loop + backstops → `RunResult`. Holds the `DOMINANT_PROMPT` / `WORKER_PROMPT` / `RESEARCH_HINT` system prompts.
- `env.py` — tiny dependency-free `.env` loader (real env vars always win).
- `config.py` — `Config` dataclass (all defaults; backward-compatible across phases).
- `cli.py` — headless entry point (see below).
- `smoke.py` — live smoke helper.

## Running

```powershell
# tests (101 passing)
.venv\Scripts\python.exe -m pytest -q

# a headless run (LM Studio must be up with models loaded)
.venv\Scripts\python.exe -m orchestrator.cli "<goal>" .\scratch --dominant 9b --worker 4b
```

- `--dominant`/`--worker` are id substrings; without them, roles fall to LM Studio load order (arbitrary — pin them).
- Tests run on Windows via the venv interpreter; `asyncio_mode = "auto"`.

## Secrets & config

- `.env` (gitignored; `.env.example` committed) holds `GEMINI_API_KEY` and `LMSTUDIO_TOKEN`. `cli.py` calls `load_dotenv()` at startup. Real OS env vars take precedence.
- `LMSTUDIO_TOKEN` is required on **all** LM Studio endpoints once its API auth is on (sent as Bearer on the OpenAI-compat loop too, not just research).
- Research is enabled **only when** `LMSTUDIO_TOKEN` is set **and** `mcp.json` (`~/.lmstudio/mcp.json` by default) lists ≥1 server; otherwise the worker runs file/command only and everything else is unchanged.

## Status & phasing

- **Phase 1** (single-agent text-protocol loop), **Phase 2** (planner + dominant/worker orchestration + `delegate` + backstops), **Phase 3** (MCP research) — all **implemented and merged**. 101 tests green.
- **Phase 4 — web UI is NOT built.** The top-level design references `server.py` (FastAPI REST + `/ws` WebSocket) and `static/` (single-page dashboard + kill switch); neither exists yet, and `fastapi`/`uvicorn` are not yet dependencies (`pyproject.toml` lists only `httpx`). Today the only entry point is the headless `cli.py`. Backstops replace the kill switch for now: max-turns cap + no-progress detector + per-worker `max_steps`.

## Docs

- Specs: `docs/superpowers/specs/` — top-level design (`2026-06-15-dual-gpu-orchestrator-design.md`) + per-phase designs.
- Plans: `docs/superpowers/plans/` — task-by-task implementation plans (phases 1–3).
- Reference: `docs/reference/lmstudio-mcp-via-api.md`, `docs/reference/lmstudio-rest-api.md` — the empirical LM Studio API findings that drove the Phase 3 native-endpoint decision.

## Conventions

- Python 3.11+, `httpx` for all HTTP (no provider SDKs), `pytest` + `pytest-asyncio`.
- Each module has one responsibility and is independently unit-tested; integration uses a **mock LM Studio** returning scripted `::action` text — the engine is built and verified without a real model. Keep that pattern.
- Never hard-code secrets; never commit `.env`.
