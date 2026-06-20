# Dual-GPU Orchestrator

A local, fully autonomous **dual-agent engine** with a live web dashboard. Two
models served by [LM Studio](https://lmstudio.ai/) — a **dominant** orchestrator
and a **worker** — cooperate to drive a goal to completion with **no human input
after kickoff**. The dominant plays the seat a human normally occupies in an agent
workflow (turn a goal into a plan, delegate, review, decide when it's done); the
worker executes one subtask at a time. The only human touchpoint during a run is
an optional kill switch.

This program is a **client + orchestrator + tool host** — it does **not** load
models or manage GPUs. LM Studio already solves model loading and device placement
across a heterogeneous GPU setup (the genuinely hard part); this engine talks to
its OpenAI-compatible API.

## Why it exists

Small local models are unreliable at native OpenAI tool-calling. So instead of the
`tools` API, every agent action flows through a forgiving **structured text
protocol** the engine parses (`::action … ::end` → `::result … ::end`), one action
per turn. That single decision is what lets weak/quantized local models drive real
file and command work reliably.

## How it works

A goal runs through a three-tier pipeline, fully autonomously:

| Tier | Who | Job |
|------|-----|-----|
| **Planner** | a frontier model (Gemini) *or* the local dominant | Goal → an ordered checklist |
| **Dominant** | local large model | Drive the checklist: delegate each step, review the worker's report, mark done, complete. Never touches files. |
| **Worker** | local mid model | Execute one subtask at a time in a **fresh** context, using first-party file/command tools |

- **First-party, sandboxed tools.** `read_file` / `write_file` / `list_dir` /
  `run_command`, all confined to a single project folder. `run_command` runs with
  the project folder as its working directory.
- **MCP research (optional).** A worker `research` verb is backed by LM Studio's
  native `/api/v1/chat` endpoint, which runs the MCP servers from your `mcp.json`
  (e.g. Exa web search) **server-side** — the engine writes no MCP protocol code.
- **Autonomy backstops.** Termination is primarily the dominant emitting
  `task_complete`; backstops are a max-turns cap, a no-progress detector, and a
  per-worker step cap, so a confused run always ends on its own.

### Reference rig

Same hardware can host surprisingly large models thanks to MoE + ternary quant:

| Role | Model | GPU | Notes |
|------|-------|-----|-------|
| Dominant | `qwen3.5-122b-a10b` (122B MoE, ~10B active) | RTX 5070 (12 GB) / CUDA | runs with **speculative decoding** using the 27B as the draft model |
| Worker | `qwen3.6-27b` (ternary-quant) | RX 7600 (8 GB) / Vulkan-ROCm | also serves as the dominant's speculative-decoding draft model |

The engine is agnostic to model size; roles are pinned by id substring.

## Requirements

- Python 3.11+
- [LM Studio](https://lmstudio.ai/) running locally with at least two models loaded
  (default API at `http://localhost:1234`)
- For MCP research: LM Studio 0.4.0+ with "allow calling servers from `mcp.json`"
  enabled, a configured server (e.g. Exa), and an API token

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# source .venv/bin/activate && pip install -e ".[dev]" # macOS/Linux
```

## Usage

### Web dashboard

```bash
python -m orchestrator.server
```

Open `http://127.0.0.1:8000`. Pick the dominant/worker models (auto-filled from LM
Studio), set a project folder and a goal, and click **Start**. The conversation,
each action and result, and the plan checklist stream live over a WebSocket; the
**Kill** button stops a run instantly.

### Headless CLI

```bash
python -m orchestrator.cli "<goal>" ./scratch --dominant 122b --worker 27b
```

`--dominant` / `--worker` are id substrings. Without them, roles fall to LM
Studio's (arbitrary) load order — pin them so the stronger model orchestrates.

## Configuration & secrets

Copy `.env.example` to `.env` (gitignored) and fill in what you need:

```
GEMINI_API_KEY=     # only for the frontier planner
LMSTUDIO_TOKEN=     # required once LM Studio API auth is on; also enables research
```

Real environment variables always take precedence over `.env`. Research turns on
only when `LMSTUDIO_TOKEN` is set **and** `mcp.json` (default `~/.lmstudio/mcp.json`)
lists at least one server; otherwise the worker runs file/command only and
everything else is unchanged. Server host/port and other limits live in
`orchestrator/config.py`.

## Testing

```bash
.venv/Scripts/python.exe -m pytest -q
```

The suite is fast and needs no real model: integration tests drive the
orchestrator with a **mock LM Studio** that returns scripted `::action` text, so
the engine is built and verified independent of model quirks.

## Project layout

```
orchestrator/
  protocol.py          # parse ::action blocks / serialize ::result (the heart; heavily tested)
  tools.py, sandbox.py # first-party file/command tools, confined to the project folder
  llm_client.py        # async client for LM Studio's OpenAI-compatible API
  agent.py             # the generic single-agent loop (emit → act → feed result)
  plan.py              # the dominant's checklist + status transitions
  planner.py           # LocalPlanner / GeminiPlanner
  coordination.py      # the dominant's verbs: delegate, mark_done, revise_plan, task_complete
  mcp_research.py      # research via LM Studio's native /api/v1/chat (MCP server-side)
  composite_registry.py# routes `research` vs first-party tools
  orchestrator.py      # owns a run: plan → dominant loop → backstops → result
  events.py            # turn-level event layer + EventBus (live observation)
  run_manager.py       # single-run asyncio lifecycle + hard-cancel kill switch
  server.py            # FastAPI: REST + /ws WebSocket
  static/              # vanilla single-page dashboard (no build step, no CDN)
  cli.py               # headless entry point
docs/superpowers/      # design specs and implementation plans, per phase
tests/                 # pytest suite
```

## Status

Built in phases, all merged: single-agent text-protocol loop → planner + dual-agent
orchestration → MCP research → web UI (live dashboard + kill switch). See
`docs/superpowers/` for the design specs and implementation plans.

## License

No license yet — all rights reserved by the author until one is added.
