"""FastAPI dashboard over the autonomous engine.

A single RunManager owns the one active run (its asyncio task + EventBus). REST
configures/starts/stops a run; the /ws WebSocket replays the current run's
buffered events then streams live ones. build_run_factory is the production run;
it is a module attribute so tests can monkeypatch it."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from orchestrator.agent import Agent
from orchestrator.composite_registry import CompositeRegistry
from orchestrator.config import Config
from orchestrator.env import load_dotenv
from orchestrator.events import make_event
from orchestrator.llm_client import LMStudioClient
from orchestrator.mcp_research import McpResearcher, mcp_integrations
from orchestrator.orchestrator import Orchestrator, worker_prompt_for
from orchestrator.planner import GeminiPlanner, LocalPlanner
from orchestrator.run_manager import RunManager
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
cfg = Config()
manager = RunManager()


class RunParams(BaseModel):
    dominant: str
    worker: str
    project: str
    goal: str
    enable_research: bool = False
    debug: bool = False


def _build_planner(cfg: Config, client: LMStudioClient, dominant_model: str):
    if cfg.planner == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            return GeminiPlanner(api_key=key, model=cfg.gemini_model)
        if not cfg.planner_fallback_local:
            raise RuntimeError("GEMINI_API_KEY not set and fallback disabled")
    return LocalPlanner(client=client, model=dominant_model)


def build_run_factory(params: dict, cfg: Config):
    """Return an async run(bus) coroutine that wires the engine and runs it,
    emitting run_started/run_finished and closing clients in finally."""

    async def _run(bus) -> None:
        token = os.environ.get("LMSTUDIO_TOKEN", "")
        client = LMStudioClient(
            base_url=cfg.lm_studio_url, timeout=cfg.request_timeout, token=token
        )
        researcher = None
        try:
            integrations = mcp_integrations(cfg.resolved_mcp_json())
            research_on = bool(params.get("enable_research")) and bool(token) and bool(integrations)
            debug_on = bool(params.get("debug"))
            if research_on:
                researcher = McpResearcher(
                    base_url=cfg.lmstudio_native_url, token=token,
                    model=cfg.research_model or params["worker"],
                    integrations=integrations, timeout=cfg.research_timeout,
                )
            worker_prompt = worker_prompt_for(research=research_on, debug=debug_on)

            bus.emit(make_event(
                "run_started", goal=params["goal"],
                dominant=params["dominant"], worker=params["worker"],
                research=research_on, debug=debug_on,
            ))

            project = Path(params["project"])
            project.mkdir(parents=True, exist_ok=True)

            def worker_factory() -> Agent:
                tool_registry = ToolRegistry(Sandbox(project), cfg.command_timeout)
                registry = (
                    CompositeRegistry(tool_registry, researcher)
                    if researcher is not None
                    else tool_registry
                )
                return Agent(
                    client=client, registry=registry, model=params["worker"],
                    system_prompt=worker_prompt, max_steps=cfg.max_steps,
                    sink=bus, agent_label="worker",
                )

            planner = _build_planner(cfg, client, params["dominant"])
            orch = Orchestrator(
                planner=planner, worker_factory=worker_factory,
                dominant_client=client, dominant_model=params["dominant"],
                max_dominant_turns=cfg.max_dominant_turns,
                no_progress_limit=cfg.no_progress_limit, sink=bus,
            )
            result = await orch.run(params["goal"])
            bus.emit(make_event("run_finished", stopped_reason=result.stopped_reason))
        finally:
            await client.aclose()
            if researcher is not None:
                await researcher.aclose()

    return _run


@app.get("/api/models")
async def api_models():
    token = os.environ.get("LMSTUDIO_TOKEN", "")
    client = LMStudioClient(
        base_url=cfg.lm_studio_url, timeout=cfg.request_timeout, token=token
    )
    try:
        models = await client.list_models()
    finally:
        await client.aclose()
    research_available = bool(token) and bool(mcp_integrations(cfg.resolved_mcp_json()))
    return {"models": models, "research_available": research_available}


@app.post("/api/run")
async def api_run(params: RunParams):
    if manager.is_running:
        raise HTTPException(status_code=409, detail="a run is already active")
    manager.start(build_run_factory(params.model_dump(), cfg))
    return {"status": "started"}


@app.post("/api/stop")
async def api_stop():
    await manager.stop()
    return {"status": "stopped"}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    queue = manager.bus.subscribe()
    try:
        for event in manager.bus.replay():
            await websocket.send_json(event)
        while True:
            await websocket.send_json(await queue.get())
    except WebSocketDisconnect:
        pass
    finally:
        manager.bus.unsubscribe(queue)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main() -> None:
    load_dotenv()
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    main()
