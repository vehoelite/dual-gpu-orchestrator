import asyncio

from orchestrator.events import make_event
from orchestrator.run_manager import RunManager


async def test_start_runs_factory_to_completion():
    mgr = RunManager()

    async def factory(bus):
        bus.emit(make_event("hello"))

    mgr.start(factory)
    await mgr.task  # wait for completion
    assert mgr.is_running is False
    assert [e["type"] for e in mgr.bus.replay()] == ["hello"]


async def test_start_resets_buffer():
    mgr = RunManager()
    mgr.bus.emit(make_event("stale"))

    async def factory(bus):
        bus.emit(make_event("fresh"))

    mgr.start(factory)
    await mgr.task
    assert [e["type"] for e in mgr.bus.replay()] == ["fresh"]


async def test_stop_emits_run_aborted():
    mgr = RunManager()

    async def factory(bus):
        await asyncio.sleep(10)

    mgr.start(factory)
    await asyncio.sleep(0)  # let the task start
    await mgr.stop()
    assert mgr.is_running is False
    assert any(e["type"] == "run_aborted" for e in mgr.bus.replay())


async def test_factory_exception_emits_error():
    mgr = RunManager()

    async def factory(bus):
        raise RuntimeError("boom")

    mgr.start(factory)
    await mgr.task
    err = [e for e in mgr.bus.replay() if e["type"] == "error"]
    assert err and "boom" in err[0]["message"]
