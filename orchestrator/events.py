"""Live-observation layer: turn-level events plus an async fan-out bus.

Events are plain JSON-serializable dicts. NullSink is the default no-op sink so
instrumentation is invisible when no UI is attached (keeps the headless engine
and the existing tests unchanged)."""
from __future__ import annotations

import asyncio
import time
from collections import deque


def make_event(event_type: str, **fields) -> dict:
    """Build an event dict with a type and a timestamp."""
    return {"type": event_type, "ts": time.time(), **fields}


def preview(text: str, cap: int = 4000) -> str:
    """Truncate long text so event frames stay small."""
    text = "" if text is None else str(text)
    return text[:cap]


def plan_event(plan) -> dict:
    """Snapshot a Plan (duck-typed on .steps) as a 'plan' event."""
    steps = [
        {"index": i, "description": s.description, "status": s.status}
        for i, s in enumerate(plan.steps)
    ]
    done = sum(1 for s in plan.steps if s.status == "done")
    return make_event("plan", steps=steps, done=done, total=len(plan.steps))


class NullSink:
    """A sink that discards events. Default everywhere instrumentation exists."""

    def emit(self, event: dict) -> None:
        return None


class EventBus:
    """Synchronous, non-blocking fan-out to per-subscriber queues, plus a bounded
    ring buffer of the current run's events for replay on (re)connect."""

    def __init__(self, queue_size: int = 1000, buffer_size: int = 2000) -> None:
        self._queue_size = queue_size
        self._subscribers: set[asyncio.Queue] = set()
        self._buffer: deque[dict] = deque(maxlen=buffer_size)

    def emit(self, event: dict) -> None:
        self._buffer.append(event)
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow consumer: drop rather than block the run

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    def replay(self) -> list[dict]:
        return list(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        # Also clear all subscriber queues
        for q in self._subscribers:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
