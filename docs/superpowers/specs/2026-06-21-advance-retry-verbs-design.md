# Lever A: Compound `advance` + context-preserving `retry` verbs

**Date:** 2026-06-21
**Status:** Approved (brainstorming) → implementation
**Goal:** Cut the dominant's generated tokens per step. On this rig the dominant
runs at ~2 t/s and output tokens cost ~5x input in wall-time, so fewer dominant
turns is the speed lever. A full primes run today is ~7m40s / peak 820 prompt
tokens across ~2K+1 dominant turns for a K-step plan.

## Problem

The current happy-path loop is **two dominant turns per step**:

```
delegate 0  ->  (read report) mark_done 0  ->  delegate 1  ->  mark_done 1  -> ... task_complete
```

A K-step plan costs ~2K+1 dominant turns. Each `mark_done` is a whole extra
round-trip (generation + prompt-eval of the growing transcript) that produces
almost no useful output. The failure path is worse: re-delegating a failed step
forces the dominant to re-author the entire corrected task body.

## Design

Two new compound verbs on `CoordinationRegistry`. All existing verbs
(`delegate`, `mark_done`, `revise_plan`, `set_plan`, `task_complete`) stay as
fallbacks. The parser (`protocol.py`) needs **no change** — these are just verbs.

### `advance` — happy path (mark current done + delegate next, one turn)

```
::action advance
done: 0
step: 1
---
<full self-contained instructions for step 1>
::end
```

Engine (`_advance`):
1. Parse + validate `done` index, `step` index, and non-empty body **before any
   mutation** (so a bad block leaves the plan untouched — same discipline as
   `_delegate` today).
2. `plan.mark_done(done)`.
3. Clear the just-completed step's active worker state.
4. `plan.mark_in_progress(step)`, build a **fresh** worker, run it, return its
   report + `plan.render()` (same shape as `delegate`'s result).

Collapses the loop to ~K+2 dominant turns: `delegate 0`, then one `advance` per
remaining step, then `mark_done` (last step) + `task_complete`.

### `retry` — failure path (same step, context preserved)

```
::action retry
step: 1
---
The price field was missing; add it and re-run.
::end
```

Engine (`_retry`):
1. Require an active worker transcript for `step` (else `error`: "no active
   worker for step N to retry").
2. **Resume** that transcript via `Agent.resume(prior_transcript, followup)`,
   where `followup` is a `::result error` carrying a fixed banner +
   the dominant's one-line note:
   > Reviewer rejected your previous attempt. Do NOT repeat it — take a
   > different approach. <dominant note>
3. Step stays `in_progress`. Return the new report.

Because the worker resumes its own transcript, it already has the original task
and its failed attempt in context — so the dominant re-states **nothing** and
the engine re-pastes **nothing**. The dominant spends ~1 line.

## Context model (the key decision)

- **Each new step → fresh worker** (preserves the cross-step isolation invariant:
  the dominant must still paste literal content — URLs, values, file contents —
  into a `delegate`/`advance` body for a *new* step, because a new worker shares
  no memory).
- **Retry of the same step → continue that worker's transcript**, discarded once
  the step is marked done (by `advance` or `mark_done`). 16k worker context is
  ample: observed peak <1K tokens; a failed attempt + a retry or two is ~2–3k.

This sharpens, not breaks, the "fresh worker per subtask" rule: a retry is the
*same* subtask, so continuing its context is consistent.

## Engine state (`CoordinationRegistry`)

- `self._active_step: int | None`
- `self._active_transcript: list[dict] | None`

Set by `delegate`/`advance` (after a fresh worker runs). Consumed by `retry`
(resume). Cleared when the owning step is marked done.

## Backstop interaction (intentional)

`execute()` already increments `no_progress_count` when the plan signature is
unchanged, and resets it on change.

- `advance` changes the signature (done + in_progress) → **resets** the counter.
- `retry` leaves the signature unchanged (step stays `in_progress`) →
  **increments** the counter. This is desired: repeated failing retries trip the
  `no_progress_limit` (5) backstop and end the run instead of looping forever.
  One retry between advances is harmless — the next `advance` resets the count.
- Each individual worker attempt is still capped by the worker's `max_steps`.

## `Agent.resume`

New method, minimal surface: the same loop as `run`, but seeded with an existing
`messages` list plus one appended user turn (the `::result error` followup),
instead of `[system, user(task)]`. `run` can delegate to a shared `_loop(messages)`.

## Events

- `advance`: existing `worker_started` / `worker_finished` + one `plan_event`
  (full state carries both transitions). No new type.
- `retry`: `worker_started` / `worker_finished` with a `retry: true` flag so the
  dashboard can render "↻ retrying step N".

## `DOMINANT_PROMPT` rewrite

Lead with the eco loop so it is what the model sees first:

1. `delegate` step 0 (full self-contained instructions).
2. Read the report; then either **`advance`** (success: done current + delegate
   next) or **`retry`** (failure: same step, one-line correction — the worker
   remembers the rest).
3. Last step: `mark_done` then `task_complete`.

Keep the CRITICAL "paste literal content into a **new** delegate/advance body"
rule (still applies for new steps), and add one line: retry is the exception —
don't re-paste, the worker remembers. `delegate` / `mark_done` / `revise_plan` /
`task_complete` examples remain below the main loop as fallbacks.

## Testing

- **`advance`** (coordination): atomic done+delegate; validation-before-mutation
  (bad `done`/`step`/empty body mutates nothing); builds a fresh worker; result
  shape matches `delegate`; resets `no_progress`.
- **`retry`** (coordination): resumes the active transcript; injects the
  rejection `::result error` with the dominant's note; step stays `in_progress`;
  increments `no_progress`; `error` when no active worker for the step.
- **`Agent.resume`**: continues from a prior `messages` list + the followup turn;
  honors `terminal_verbs` / `max_steps`.
- Parser: no change (verbs only) — assert `advance`/`retry` parse via existing
  tests if convenient.

## Out of scope (YAGNI)

- No `redo` flag folded into `advance` (rejected: overloaded semantics).
- No auto-advance / engine-driven multi-step batching (removes the dominant's
  per-step review).
- `task_complete` stays pure (no implicit final `mark_done`).
