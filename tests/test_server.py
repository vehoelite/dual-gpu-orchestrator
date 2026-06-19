from fastapi.testclient import TestClient

from orchestrator import server
from orchestrator.events import make_event
from orchestrator.run_manager import RunManager

_BODY = {"dominant": "d", "worker": "w", "project": "./scratch", "goal": "g"}


def test_api_models(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def list_models(self):
            return ["m1", "m2"]

        async def aclose(self):
            pass

    monkeypatch.setattr(server, "LMStudioClient", FakeClient)
    monkeypatch.setattr(server, "mcp_integrations", lambda path: [])
    monkeypatch.delenv("LMSTUDIO_TOKEN", raising=False)
    with TestClient(server.app) as client:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"] == ["m1", "m2"]
        assert body["research_available"] is False


def test_run_streams_events_over_ws(monkeypatch):
    server.manager = RunManager()

    async def fake_factory(bus):
        bus.emit(make_event("run_started", goal="g", dominant="d", worker="w", research=False))
        bus.emit(make_event("run_finished", stopped_reason="task_complete"))

    monkeypatch.setattr(server, "build_run_factory", lambda params, cfg: fake_factory)
    with TestClient(server.app) as client:
        with client.websocket_connect("/ws") as ws:
            assert client.post("/api/run", json=_BODY).status_code == 200
            received = [ws.receive_json()["type"] for _ in range(2)]
        assert "run_started" in received
        assert "run_finished" in received


def test_double_run_returns_409(monkeypatch):
    import asyncio

    server.manager = RunManager()

    async def long_factory(bus):
        await asyncio.sleep(5)

    monkeypatch.setattr(server, "build_run_factory", lambda params, cfg: long_factory)
    with TestClient(server.app) as client:
        assert client.post("/api/run", json=_BODY).status_code == 200
        assert client.post("/api/run", json=_BODY).status_code == 409
        client.post("/api/stop")


def test_stop_emits_run_aborted(monkeypatch):
    import asyncio

    server.manager = RunManager()

    async def long_factory(bus):
        await asyncio.sleep(5)

    monkeypatch.setattr(server, "build_run_factory", lambda params, cfg: long_factory)
    with TestClient(server.app) as client:
        client.post("/api/run", json=_BODY)
        assert client.post("/api/stop").status_code == 200
        assert any(e["type"] == "run_aborted" for e in server.manager.bus.replay())
