# Phase 2: Dual-Agent Orchestration — Design

**Date:** 2026-06-14
**Status:** Approved (design phase)
**Builds on:** Phase 1 (`docs/superpowers/specs/2026-06-15-dual-gpu-orchestrator-design.md`, `docs/superpowers/plans/2026-06-14-phase1-single-agent-loop.md`)

## 1. Summary

Phase 2 makes the engine autonomous and dual-agent. A run takes a goal and drives
it to completion with **no human input after kickoff** (the project's core
invariant). Three tiers cooperate:

| Tier | Who | Job |
|------|-----|-----|
| **Planner** | frontier API *or* local 9B | Goal → an ordered checklist |
| **Dominant** | local 9B | Drive the checklist: delegate, review, mark done, complete |
| **Worker** | local 4B (Phase 1 `Agent`) | Execute one subtask at a time in a fresh context |

**Design rationale.** The plan is the highest-leverage artifact in a run: a good
decomposition lets a weak 4B worker succeed; a bad one dooms the run. So planning
is a pluggable step that can use a high-quality frontier model, while execution
stays local (private, free, offline-capable). The dominant therefore is a pure
**orchestrator/reviewer** — it does not plan and does not touch the filesystem.

Phase 2 is **orchestration only**. External tools (`mcp_host`/Exa) are Phase 3;
the web UI + kill switch are Phase 4.

## 2. Goals / Non-Goals

**Goals**
- Turn a goal into a checklist via a pluggable `Planner` (frontier or local).
- Drive the checklist to completion autonomously: dominant delegates subtasks to
  a fresh worker, reviews results, marks steps done, revises when needed.
- Reuse the Phase 1 `Agent` loop for both worker and dominant.
- Run fully headless with automatic termination/backstops (no human, no UI yet).
- Keep the system able to run 100% local/offline (LocalPlanner fallback).

**Non-Goals**
- No external tools / MCP (`web_search`/Exa) — Phase 3.
- No web UI, WebSocket streaming, or interactive kill switch — Phase 4.
- No native OpenAI tool-calling — all agent actions remain text-protocol (Phase 1).
- No per-action human approval — anything is allowed inside the project folder.
- No multi-worker parallelism — one worker subtask at a time.

## 3. Architecture

New files (each one clear responsibility, independently testable):

- **`planner.py`** — `Planner` protocol with `async make_plan(goal: str) -> list[str]`
  returning ordered step descriptions. Implementations:
  - `LocalPlanner(client, model)` — prompts the dominant model via the existing
    `LMStudioClient` to emit a checklist; parses it into steps.
  - `GeminiPlanner(api_key, model)` — calls Google's Gemini API. Reads its key
    from the `GEMINI_API_KEY` environment variable (never hard-coded, never
    committed). HTTP via `httpx`.
- **`plan.py`** — `Plan`: an ordered list of `Step(description, status)` where
  status ∈ `pending | in_progress | done`. Methods: `mark_done(index)`,
  `revise(new_steps)`, `all_done() -> bool`, `render() -> str` (human/model-
  readable status block). Pure data + transitions, no I/O.
- **`coordination.py`** — `CoordinationRegistry`, mirroring `ToolRegistry`'s
  interface `execute(action: Action) -> tuple[str, str]`. Handles the dominant's
  control verbs (see §5). `delegate` builds and runs a fresh worker `Agent` and
  returns its outcome as the `(status, message)`. Holds references to the `Plan`,
  a worker-`Agent` factory, and an event sink.
- **`orchestrator.py`** — `Orchestrator.run(goal) -> RunResult`. Owns a run:
  calls the planner, seeds the `Plan`, runs the dominant loop (a Phase 1 `Agent`
  wired to a `CoordinationRegistry`), enforces backstops, emits events.

Reused from Phase 1 (unchanged except one small extension, §4): `agent.Agent`,
`tools.ToolRegistry`, `protocol`, `sandbox`, `llm_client`, `config`.

## 4. Reuse of the Phase 1 Agent loop

The Phase 1 `Agent` loop is generic over a registry exposing
`execute(action) -> (status, message)`. Phase 2 exploits this:

- **Worker** = `Agent` + `ToolRegistry` (exactly Phase 1).
- **Dominant** = the same `Agent` + a `CoordinationRegistry`.

**One required Phase 1 extension:** the `Agent`'s terminal verb is currently the
hard-coded `"done"`. Make it configurable (`terminal_verbs: set[str]`, default
`{"done"}`). The worker keeps `{"done"}`; the dominant uses `{"task_complete"}`.
This is a backward-compatible change (default preserves Phase 1 behavior and
tests).

**Keeping the dominant aware of plan state.** After every coordination action,
the returned `::result` message includes the re-rendered `Plan.render()` so the
dominant always sees current status (which steps are pending/done) without us
leaking the worker's history or other extra context. The dominant's initial user
message contains the goal and the full checklist.

## 5. Coordination verbs (dominant only)

All via the existing text protocol (`::action <verb> … ::end`). The dominant has
ONLY these verbs (no file/command verbs — it never touches the filesystem):

| Verb | Args / body | Effect |
|------|-------------|--------|
| `set_plan` | body = checklist lines | (Re)initialize the `Plan`. Optional — the planner usually seeds it; allows the dominant to restate. |
| `delegate` | `step` (index), body = subtask + context | Run a fresh worker `Agent` on the subtask; return its result. Marks the step `in_progress`. |
| `mark_done` | `step` (index) | Mark a step `done`. |
| `revise_plan` | body = new checklist lines | Replace remaining steps when reality diverges. |
| `task_complete` | — | Terminal: end the run successfully. |

Unknown/blank verbs and args follow Phase 1 handling: `CoordinationRegistry`
returns `("error", message)`; malformed blocks get the corrective reprompt.

## 6. Delegation & fresh worker context

When the dominant emits `delegate`, the `CoordinationRegistry`:
1. Marks the target step `in_progress`.
2. Builds a **fresh** worker `Agent` (new conversation, worker system prompt +
   `ToolRegistry` over the same project sandbox).
3. Runs `await worker.run(subtask_with_context)` where the task text is the
   delegate body (subtask + the context string the dominant chose to pass).
   **The dominant's conversation is never shared with the worker** — this mirrors
   subagent-driven development (fresh subagent per task).
4. Returns `(status, message)` to the dominant: `message` summarizes the worker's
   outcome — its `stopped_reason` plus its final assistant message (its
   "report") — followed by the re-rendered plan. The dominant then decides
   `mark_done`, `revise_plan`, or re-`delegate`.

## 7. Termination & autonomy

Primary termination: dominant emits `task_complete` (intended once `all_done()`).
Automatic backstops (no human required, upholding the core invariant):
- **max dominant-turns** cap on the dominant loop.
- **no-progress detector**: abort if N consecutive dominant turns produce no
  plan-state change (no `mark_done`/`revise_plan`/successful `delegate`).
- **per-worker `max_steps`**: already enforced by the Phase 1 `Agent`.

`Orchestrator.run` returns a `RunResult` capturing final plan state, the dominant
transcript, per-delegation worker results, and a `stopped_reason`
(`task_complete | max_turns | no_progress | planner_failed`).

## 8. Error handling

| Scenario | Handling |
|----------|----------|
| Planner returns empty/garbled checklist | Orchestrator validates; retries the planner once; else aborts run with `stopped_reason="planner_failed"`. |
| `GeminiPlanner` network/key error | Raised at run start (before the loop). Config flag `planner_fallback_local` can auto-fall back to `LocalPlanner`. |
| Worker subtask fails (`max_steps`/`no_action`) | Reported back to the dominant as the delegate result; dominant decides revise/retry. Never crashes the run. |
| Malformed dominant action | Phase 1 corrective-reprompt path (counts as a turn). |
| Unknown coordination verb / bad step index | `("error", message)` returned; dominant retries next turn. |
| Runaway / stuck | max-turns + no-progress backstops stop the run. |

## 9. Configuration additions (`config.py`)

Extend the Phase 1 `Config` (backward compatible defaults):
- `planner: str = "local"` — `"local"` or `"gemini"`.
- `gemini_model: str = "gemini-2.0-flash"` (key from `GEMINI_API_KEY` env, not config).
- `planner_fallback_local: bool = True`.
- `max_dominant_turns: int = 40`.
- `no_progress_limit: int = 5`.

## 10. Testing

- **`plan.py`** — unit: status transitions, `revise`, `all_done`, `render`.
- **`coordination.py`** — unit with a **mock worker factory**: `delegate` routes
  the subtask and returns its result; `mark_done`/`revise_plan` mutate the plan;
  unknown verb/bad index → error; `task_complete` recognized.
- **`planner.py`** — `LocalPlanner` with a fake `LMStudioClient` (scripted
  checklist text → parsed steps); `GeminiPlanner` with a mocked `httpx`
  transport (no live key in tests) asserting request shape + response parsing.
- **`orchestrator.py`** — deterministic integration via the **mock LM Studio**
  (scripted dominant actions) + a stub planner: full goal → plan → delegate →
  worker (also scripted) → `mark_done` → `task_complete`. Verifies a complete
  autonomous run with no real models. Also cover the no-progress and max-turns
  backstops.
- **Optional live smoke** (excluded from the unit suite): real LM Studio
  (dominant+worker) + real Gemini planner on a trivial goal.

## 11. Stack & Phasing

**Stack:** unchanged from Phase 1 — Python 3.11+, `httpx` (async), `pytest` +
`pytest-asyncio`. Gemini via `httpx` (no provider SDK dependency).

**Build order (single plan):**
1. `plan.py` (pure data) → 2. extend `Agent` terminal verbs → 3. `planner.py`
   (Local + Gemini) → 4. `coordination.py` (delegate + plan verbs) →
   5. `orchestrator.py` (wire planner + dominant loop + backstops) → 6. config +
   headless run entry point + optional live smoke.

## 12. Risks

- **Dominant coordination reliability.** The 9B must reliably emit coordination
  actions and judge worker results. Mitigations: the forgiving text protocol,
  always re-showing plan status, corrective reprompts, and no-progress/max-turns
  backstops so a confused dominant still terminates.
- **Plan↔execution mismatch.** A frontier plan may assume capabilities the local
  worker lacks. Mitigation: `revise_plan`, plus the dominant reviewing each
  worker result before marking done.
- **Gemini key/format uncertainty.** The provided key has an unusual prefix;
  validated at integration time. The mocked tests don't depend on it, and
  `planner_fallback_local` keeps runs working if the frontier path fails.
