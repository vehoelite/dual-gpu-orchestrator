"""Tiny dependency-free .env loader so users can keep secrets in a file instead
of setting OS environment variables by hand. Real environment variables always
take precedence (we use setdefault)."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env", environ: dict | None = None) -> dict[str, str]:
    """Load KEY=VALUE pairs from ``path`` into ``environ`` (defaults to
    os.environ) without overriding existing values. Returns the pairs found in
    the file. Missing file is a no-op."""
    env = os.environ if environ is None else environ
    loaded: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        env.setdefault(key, value)
        loaded[key] = value
    return loaded
