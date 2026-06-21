from fastapi.testclient import TestClient

from orchestrator import server
from orchestrator.events import make_event
from orchestrator.run_manager import RunManager

_BODY = {"dominant": "d", "worker": "w", "project": "./scratch", "goal": "g"}


def test_run_params_debug_defaults_false():
    assert server.RunParams(**_BODY).debug is False


def test_run_params_accepts_debug():
    assert server.RunParams(**{**_BODY, "debug": True}).debug is True


def test_run_params_planner_defaults_local():
    p = server.RunParams(**_BODY)
    assert p.planner == "local"
    assert p.planner_model == ""


def test_build_planner_local_by_default(monkeypatch):
    from orchestrator.planner import LocalPlanner

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    planner = server.build_planner(server.cfg, client=None, params={**_BODY, "dominant": "dom"})
    assert isinstance(planner, LocalPlanner)
    assert planner.model == "dom"


def test_build_planner_local_when_gemini_requested_but_no_key(monkeypatch):
    from orchestrator.planner import LocalPlanner

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    params = {**_BODY, "dominant": "dom", "planner": "gemini"}
    planner = server.build_planner(server.cfg, client=None, params=params)
    assert isinstance(planner, LocalPlanner)  # falls back (planner_fallback_local)


async def test_build_planner_gemini_when_selected_with_key(monkeypatch):
    from orchestrator.planner import GeminiPlanner

    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    params = {**_BODY, "dominant": "dom", "planner": "gemini", "planner_model": "gemini-3.5-flash"}
    planner = server.build_planner(server.cfg, client=None, params=params)
    assert isinstance(planner, GeminiPlanner)
    assert planner.model == "gemini-3.5-flash"
    await planner.aclose()


async def test_build_planner_gemini_defaults_model_from_cfg(monkeypatch):
    from orchestrator.planner import GeminiPlanner

    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    params = {**_BODY, "dominant": "dom", "planner": "gemini"}
    planner = server.build_planner(server.cfg, client=None, params=params)
    assert isinstance(planner, GeminiPlanner)
    assert planner.model == server.cfg.gemini_model
    await planner.aclose()


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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with TestClient(server.app) as client:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"] == ["m1", "m2"]
        assert body["research_available"] is False
        assert body["premium_planner_available"] is False


def test_api_models_premium_planner_available(monkeypatch):
    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def list_models(self):
            return ["m1"]

        async def aclose(self):
            pass

    monkeypatch.setattr(server, "LMStudioClient", FakeClient)
    monkeypatch.setattr(server, "mcp_integrations", lambda path: [])
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    with TestClient(server.app) as client:
        body = client.get("/api/models").json()
        assert body["premium_planner_available"] is True


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
