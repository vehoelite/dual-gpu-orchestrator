"""Owns the single active run: an asyncio task wrapping a run-factory, plus the
EventBus that streams its events. Hard-cancel kills the task; cleanup and the
terminal event live here so the server stays thin."""
from __future__ import annotations

import asyncio
import contextlib

from orchestrator.events import EventBus, make_event


class RunManager:
    def __init__(self) -> None:
        self.bus = EventBus()
        self.task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self.task is not None and not self.task.done()

    def start(self, factory) -> None:
        """factory: async def (bus: EventBus) -> None. Resets the bus buffer and
        launches the run as a background task."""
        self.bus.reset()
        self.task = asyncio.create_task(self._guard(factory))

    async def _guard(self, factory) -> None:
        try:
            await factory(self.bus)
        except asyncio.CancelledError:
            self.bus.emit(make_event("run_aborted"))
            raise
        except Exception as exc:  # surface, don't crash the server
            self.bus.emit(make_event("error", message=str(exc)))

    async def stop(self) -> None:
        if self.is_running:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
