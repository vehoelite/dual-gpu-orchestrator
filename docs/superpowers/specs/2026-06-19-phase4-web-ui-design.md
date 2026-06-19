# Phase 4: Web UI — Design

**Date:** 2026-06-19
**Status:** Approved (design phase)
**Builds on:** Phase 1 (single-agent loop), Phase 2 (dual-agent orchestration), and Phase 3 (MCP research), all merged.

## 1. Summary

Phase 4 adds a live web dashboard over the existing autonomous engine. From a
single page the user configures a run (pick dominant/worker models, set the
project folder + goal, see whether research is available), starts it, watches the
conversation, actions, and plan checklist update **live**, and can **kill** the
run at any moment. After kickoff the run is still fully autonomous (the core
invariant); the UI only observes and offers the one allowed human touchpoint — the
kill switch.

The work is mostly **backend instrumentation**. Today the engine runs synchronously
to completion and returns a `RunResult`: `agent.py` loops internally with no
per-turn callback, `coordination.py` accumulates worker results but emits nothing
live, `orchestrator.py` only returns at the end, and there is no cancellation hook.
Phase 4 threads an **optional event sink** through those loops (default off, so the
headless `cli.py` and all 101 existing tests are unchanged), then exposes the
stream over FastAPI + WebSocket with a vanilla single-page frontend.

**Granularity decision:** events are **turn-level**, not token-level. `llm_client`
stays `stream: False` and untouched — an event fires as each turn completes
(assistant message, parsed action, result, plan change, worker start/finish, run
end). This keeps the most heavily-tested module out of scope and avoids
mid-generation protocol-parsing complexity.

## 2. Goals / Non-Goals

**Goals**
- A single-page dashboard: model/role selection (auto-filled from LM Studio),
  project-folder + goal inputs, a research availability indicator + toggle, a live
  conversation/action log, a live plan checklist, and a kill switch.
- Stream turn-level run events to the browser over a WebSocket.
- A late-joining or refreshed browser replays the current run from the start, then
  continues live (ring buffer of the active run's events).
- A kill switch that stops a run **immediately** (hard asyncio cancellation,
  interrupting even an in-flight generation) with clean shutdown.
- Thread all of the above through the engine **backward-compatibly**: the event
  sink and cancellation are optional; `cli.py` and existing tests run unchanged.

**Non-Goals**
- No token-level streaming (`llm_client` stays `stream: False`).
- No multiple concurrent runs — exactly one active run at a time.
- No run history / persistence of finished runs (could be a later phase).
- No mid-run human steering (editing the plan, injecting guidance) — that would
  break the "no human in the loop after kickoff" invariant.
- No auth / multi-user / remote concerns (local single-user tool; binds to
  `127.0.0.1`).
- No frontend build step or CDN dependency (plain HTML/CSS/JS, offline-capable).

## 3. Architecture

**New files**

- **`events.py`** — the live-observation layer.
  - Event payloads are small JSON-serializable dicts, each with a `type` and a
    `ts` (epoch seconds). Helper constructors keep shapes consistent.
  - `EventBus`: an async fan-out. `subscribe() -> asyncio.Queue` /
    `unsubscribe(queue)`; `emit(event: dict)` appends to a bounded **ring buffer**
    (the current run's history, for replay) and pushes to every subscriber queue;
    `replay() -> list[dict]` returns the buffer; `reset()` clears it at the start
    of a new run. Implements the sink interface (`emit`).
  - `NullSink`: a no-op `emit` used as the default everywhere, so instrumentation
    is invisible when no UI is attached.
- **`server.py`** — FastAPI app plus a `RunManager`.
  - `RunManager` owns the single active run: its `asyncio.Task`, its `EventBus`,
    and the per-run clients (so they can be closed on completion or cancel). It
    exposes `is_running`, `start(...)`, and `stop()`.
  - Routes:
    - `GET /` — serve the SPA; `static/` mounted for assets.
    - `GET /api/models` — proxy `LMStudioClient.list_models()`.
    - `POST /api/run` — body `{dominant, worker, project, goal, enable_research}`.
      Resets the bus, builds clients + planner + orchestrator wired to the bus,
      launches the run as a background task. Returns **409** if a run is active.
    - `POST /api/stop` — `RunManager.stop()` cancels the task (hard cancel).
    - `WS /ws` — on connect, send `replay()` then stream live events from a fresh
      subscriber queue until the socket closes.
- **`static/index.html`, `static/app.js`, `static/style.css`** — vanilla SPA, no
  build, no CDN. Renders the config form, the live log (messages/actions/results,
  labeled dominant vs worker), the plan checklist with per-step status, and the
  Start/Kill controls. Connects to `/ws`, dispatches on `event.type`.

**Modifications (all optional/default-off, backward-compatible)**

- **`agent.py`** — add optional `sink` and an `agent_label` (e.g. `"dominant"` /
  `"worker"`). Emit `message` (assistant reply), `action` (parsed verb + args +
  short body preview), `result` (status + message preview), and `parse_error`.
  Default `sink=NullSink()` preserves Phase 1 behavior.
- **`coordination.py`** — add optional `sink`. Emit `worker_started` /
  `worker_finished` around each delegation and a `plan` event whenever plan state
  changes (`set_plan`/`revise_plan`/`mark_done`/successful `delegate`), plus
  `no_progress` when the backstop trips.
- **`orchestrator.py`** — add optional `sink`. Emit `run_started`, an initial
  `plan` (after seeding), and `run_finished` (with `stopped_reason`). Wire the sink
  into the dominant `Agent` and the `CoordinationRegistry`; the worker_factory
  injects the sink (and `agent_label="worker"`) into each worker so worker turns
  stream too. Hard cancellation needs no special logic here — `task.cancel()`
  raises `CancelledError` out of the in-flight `await client.complete`; the server
  catches it, emits `run_aborted`, and closes clients in `finally`.
- **`config.py`** — add `host: str = "127.0.0.1"` and `port: int = 8000`.
- **`pyproject.toml`** — add `fastapi` and `uvicorn[standard]` dependencies; add a
  `python -m orchestrator.server` entry point that runs uvicorn on `cfg.host:port`.
- **`cli.py`** — unchanged (passes no sink; runs exactly as in Phase 3).

## 4. Event protocol

Each event is a dict with `type` + `ts` and type-specific fields:

| `type` | Fields | Emitted by |
|--------|--------|------------|
| `run_started` | `goal, dominant, worker, research` (bool) | orchestrator |
| `plan` | `steps: [{index, description, status}], done, total` | orchestrator (seed), coordination (on change) |
| `message` | `agent, text` | agent |
| `action` | `agent, verb, args, body_preview` | agent |
| `result` | `agent, status, message_preview` | agent |
| `parse_error` | `agent, error` | agent |
| `worker_started` | `step, subtask` | coordination |
| `worker_finished` | `step, stopped_reason` | coordination |
| `no_progress` | — | coordination |
| `run_finished` | `stopped_reason` | server (after orchestrator returns) |
| `run_aborted` | — | server (on cancel) |
| `error` | `message` | server (run-start failures) |

Long text (`text`, `message_preview`, `body_preview`) is truncated to a sane cap
before emission to keep frames small.

## 5. Control flow

1. SPA loads → `GET /api/models` fills the dominant/worker dropdowns. The research
   indicator reflects whether `LMSTUDIO_TOKEN` + `mcp.json` servers are available.
2. User sets project folder + goal, picks models, optionally toggles research off,
   clicks Start → `POST /api/run`. **Last human input of the run.**
3. Server resets the bus, builds the orchestrator wired to the bus, launches the
   run task, returns 200 (or 409 if one is already active).
4. The browser's `/ws` connection receives the replayed buffer then live events;
   it renders the conversation/action log and updates the checklist on each `plan`.
5. Run ends by `task_complete`/backstop → `run_finished`; or the user clicks Kill →
   `POST /api/stop` → `run_aborted`. Clients are closed either way.

## 6. Error handling

| Scenario | Handling |
|----------|----------|
| LM Studio unreachable / no models / bad model id | Caught at run start; emitted as `error`; the loop never starts (no 500 to the browser). |
| Mid-run LLM/network error | Propagates out of the run task; surfaced as `run_finished` with the error reason (the run task wraps the orchestrator call and reports rather than crashing the server). |
| Kill switch pressed | `task.cancel()`; `CancelledError` caught; `run_aborted` emitted; per-run clients closed. |
| `POST /api/run` while a run is active | HTTP 409; existing run untouched. |
| WebSocket client disconnects | Its subscriber queue is removed; the run continues unaffected. |
| Browser refresh / late join mid-run | `/ws` replays the ring buffer, then streams live. |

## 7. Testing

- **`events.py`** — unit: `subscribe`/`emit` fans out to multiple queues;
  `unsubscribe` stops delivery; ring buffer caps and `replay()` returns history;
  `reset()` clears it; `NullSink.emit` is a no-op.
- **`agent.py` / `coordination.py` / `orchestrator.py`** — extend the existing
  **mock LM Studio** integration tests with a **recording sink** and assert the
  event sequence for a full run (`run_started` → `plan` → dominant `action`
  delegate → `worker_started` → worker `message`/`action`/`result` →
  `worker_finished` → `plan` → `mark_done` → `task_complete` → `run_finished`).
  Re-run the existing no-sink tests to prove backward compatibility.
- **`server.py`** — Starlette/FastAPI `TestClient` (no real LM Studio): `GET
  /api/models` with a stubbed client; `POST /api/run` starts a run (stub
  orchestrator) and a second concurrent call returns 409; `POST /api/stop`
  cancels; a `/ws` client receives replayed + live events. Cover the run-start
  `error` path (models unavailable).
- **Optional live smoke** (outside the unit suite): launch `python -m
  orchestrator.server`, run a trivial goal against real LM Studio, watch the
  dashboard log + checklist update and the kill switch stop a run.

## 8. Stack & Phasing

**Stack:** Python 3.11+, `httpx` (existing), **`fastapi` + `uvicorn[standard]`
(new)**, `pytest` + `pytest-asyncio`. Vanilla HTML/CSS/JS frontend — no build
tooling, no CDN (offline-capable).

**Build order (single plan):**
1. `events.py` (`EventBus` + `NullSink` + event constructors) with unit tests.
2. Instrument `agent.py` (optional sink) + tests; confirm backward compat.
3. Instrument `coordination.py` + `orchestrator.py` (optional sink, worker_factory
   injection) + extend integration tests with a recording sink.
4. `config.py` host/port; `server.py` (`RunManager` + REST + `/ws`) + `TestClient`
   tests.
5. `static/` SPA (config form, live log, checklist, kill switch); wire to REST +
   WS; `python -m orchestrator.server` entry point; `pyproject.toml` deps.
6. Optional live smoke.

## 9. Risks

- **Cancellation cleanup.** A hard cancel can interrupt an in-flight request; the
  per-run clients must always be closed (in `finally`) and the bus must emit
  exactly one terminal event (`run_aborted`). Mitigation: `RunManager` centralizes
  the task + clients and owns the terminal-event/cleanup logic; covered by a
  cancel test.
- **WebSocket lifecycle.** Disconnects, refreshes, and slow consumers must not
  stall the run. Mitigation: per-subscriber queues, drop a subscriber on send
  failure, and the run never blocks on `emit` (bounded queues / non-blocking put).
- **Backward compatibility.** Instrumentation must not change headless behavior.
  Mitigation: every sink parameter defaults to `NullSink`; the existing no-sink
  test suite is re-run as the compatibility gate.
- **Frontend scope creep.** Plain-JS UIs invite ad-hoc growth. Mitigation: a tight
  event-dispatch table (`type` → render) and the explicit MVP scope in §2.
