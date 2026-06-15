"""First-party, in-process tools. All paths are sandboxed; run_command is
cwd-scoped to the project folder (not jailed — a known limitation)."""
from __future__ import annotations

import subprocess
from typing import Callable

from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox, SandboxError

ToolHandler = Callable[[Action], str]


class ToolError(Exception):
    """Raised by a tool handler when an action cannot be completed."""


class ToolRegistry:
    def __init__(self, sandbox: Sandbox, command_timeout: float = 60.0) -> None:
        self.sandbox = sandbox
        self.command_timeout = command_timeout
        self._handlers: dict[str, ToolHandler] = {
            "read_file": self._read_file,
            "write_file": self._write_file,
            "list_dir": self._list_dir,
            "run_command": self._run_command,
        }

    def execute(self, action: Action) -> tuple[str, str]:
        handler = self._handlers.get(action.verb)
        if handler is None:
            return "error", f"unknown verb: {action.verb}"
        try:
            return "ok", handler(action)
        except (ToolError, SandboxError) as exc:
            return "error", str(exc)

    def _require_arg(self, action: Action, key: str) -> str:
        value = action.args.get(key)
        if value is None:
            raise ToolError(f"{action.verb} requires arg '{key}'")
        return value

    def _read_file(self, action: Action) -> str:
        path = self.sandbox.resolve(self._require_arg(action, "path"))
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise ToolError(f"no such file: {action.args['path']}")
        except OSError as exc:
            raise ToolError(f"read failed: {exc}")

    def _write_file(self, action: Action) -> str:
        rel = self._require_arg(action, "path")
        path = self.sandbox.resolve(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = action.body
        path.write_text(data, encoding="utf-8")
        return f"wrote {len(data.encode('utf-8'))} bytes to {rel}"

    def _list_dir(self, action: Action) -> str:
        rel = action.args.get("path", ".")
        path = self.sandbox.resolve(rel)
        if not path.is_dir():
            raise ToolError(f"not a directory: {rel}")
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
        return "\n".join(entries) if entries else "(empty)"

    def _run_command(self, action: Action) -> str:
        cmd = self._require_arg(action, "cmd")
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=self.sandbox.root,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"command timed out after {self.command_timeout}s")
        output = (proc.stdout or "") + (proc.stderr or "")
        return f"exit={proc.returncode}\n{output}".rstrip()
