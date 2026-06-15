"""The project folder is the trust boundary for first-party file tools."""
from __future__ import annotations

from pathlib import Path


class SandboxError(Exception):
    """Raised when a path would resolve outside the project folder."""


class Sandbox:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def resolve(self, path: str) -> Path:
        """Resolve ``path`` (relative to root, or absolute) and ensure it stays
        inside the project folder. Raises ``SandboxError`` otherwise."""
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxError(f"path escapes project folder: {path!r}")
        return candidate
