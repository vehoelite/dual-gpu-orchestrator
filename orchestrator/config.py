"""Central configuration. Phase 1 uses a subset; later phases extend it."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_mcp_json() -> str:
    return os.path.expanduser("~/.lmstudio/mcp.json")


@dataclass
class Config:
    lm_studio_url: str = "http://localhost:1234/v1"
    mcp_json_path: str = ""
    request_timeout: float = 120.0
    command_timeout: float = 60.0
    max_steps: int = 50
    # Phase 2: orchestration
    planner: str = "local"  # "local" | "gemini"
    gemini_model: str = "gemini-2.0-flash"
    planner_fallback_local: bool = True
    max_dominant_turns: int = 40
    no_progress_limit: int = 5
    dominant_model: str = ""
    worker_model: str = ""
    # Phase 3: MCP research
    lmstudio_native_url: str = "http://localhost:1234"
    research_model: str = ""
    research_timeout: float = 180.0

    def __post_init__(self) -> None:
        if not self.mcp_json_path:
            self.mcp_json_path = _default_mcp_json()

    def resolved_mcp_json(self) -> Path:
        return Path(self.mcp_json_path)
