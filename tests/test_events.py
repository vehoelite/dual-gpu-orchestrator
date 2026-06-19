import asyncio

from orchestrator.events import EventBus, NullSink, make_event, plan_event, preview


def test_make_event_has_type_and_ts():
    ev = make_event("message", agent="worker", text="hi")
    assert ev["type"] == "message"
    assert isinstance(ev["ts"], float)
    assert ev["agent"] == "worker"
    assert ev["text"] == "hi"


def test_preview_truncates():
    assert preview("abcdef", cap=3) == "abc"
    assert preview("ab", cap=3) == "ab"


def test_plan_event_shape():
    class FakeStep:
        def __init__(self, d, s):
            self.description = d
            self.status = s

    class FakePlan:
        steps = [FakeStep("a", "done"), FakeStep("b", "pending")]

    ev = plan_event(FakePlan())
    assert ev["type"] == "plan"
    assert ev["total"] == 2
    assert ev["done"] == 1
    assert ev["steps"] == [
        {"index": 0, "description": "a", "status": "done"},
        {"index": 1, "description": "b", "status": "pending"},
    ]


async def test_nullsink_emit_is_noop():
    assert NullSink().emit(make_event("x")) is None


async def test_bus_fans_out_and_buffers():
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()
    bus.emit(make_event("a"))
    assert (await q1.get())["type"] == "a"
    assert (await q2.get())["type"] == "a"
    assert [e["type"] for e in bus.replay()] == ["a"]


async def test_bus_unsubscribe_stops_delivery():
    bus = EventBus()
    q = bus.subscribe()
    bus.unsubscribe(q)
    bus.emit(make_event("a"))
    assert q.empty()
    # but the buffer still records it
    assert len(bus.replay()) == 1


async def test_bus_reset_clears_buffer_keeps_subscribers():
    bus = EventBus()
    q = bus.subscribe()
    bus.emit(make_event("old"))
    bus.reset()
    assert bus.replay() == []
    bus.emit(make_event("new"))
    assert (await q.get())["type"] == "new"


async def test_bus_drops_for_full_slow_consumer():
    bus = EventBus(queue_size=1)
    q = bus.subscribe()
    bus.emit(make_event("a"))
    bus.emit(make_event("b"))  # q full -> dropped, must not raise
    assert q.qsize() == 1
    assert len(bus.replay()) == 2
