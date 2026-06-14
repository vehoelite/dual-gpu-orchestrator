"""Parse the structured text action protocol and serialize results.

Agents act by emitting an ``::action`` block in their normal text output:

    ::action <verb>
    key: value
    ---
    optional body
    ::end

The parser is deliberately forgiving so weak models can use it reliably.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_RE = re.compile(
    r"::action[ \t]+(?P<header>.*?)\n(?P<inner>.*?)\n?[ \t]*::end",
    re.DOTALL,
)


class ProtocolError(Exception):
    """Raised when an ``::action`` marker is present but the block is malformed."""


@dataclass(frozen=True)
class Action:
    verb: str
    args: dict[str, str]
    body: str


def parse_action(text: str) -> Action | None:
    """Return the first well-formed Action in ``text``.

    Returns ``None`` if there is no ``::action`` marker at all (the model is
    just talking). Raises ``ProtocolError`` if a marker is present but the block
    cannot be parsed (e.g. missing ``::end`` or missing verb).
    """
    text = text.replace("\r\n", "\n")
    if "::action" not in text:
        return None

    match = _ACTION_RE.search(text)
    if match is None:
        raise ProtocolError("found '::action' but no closing '::end'")

    verb = match.group("header").strip()
    if not verb:
        raise ProtocolError("action is missing a verb")

    inner = match.group("inner")
    args, body = _split_args_and_body(inner)
    return Action(verb=verb, args=args, body=body)


def _split_args_and_body(inner: str) -> tuple[dict[str, str], str]:
    args: dict[str, str] = {}
    lines = inner.split("\n")
    body_lines: list[str] | None = None

    for i, line in enumerate(lines):
        if body_lines is not None:
            body_lines.append(line)
            continue
        if line.strip() == "---":
            body_lines = []
            continue
        if line.strip() == "":
            continue
        key, sep, value = line.partition(":")
        if sep == "":
            # A non key:value line before '---' starts the body implicitly.
            body_lines = lines[i:]
            break
        args[key.strip()] = value.strip()

    body = "\n".join(body_lines).strip("\n") if body_lines is not None else ""
    return args, body


def serialize_result(status: str, message: str) -> str:
    """Render an engine -> model result block."""
    if status not in ("ok", "error"):
        raise ValueError(f"status must be 'ok' or 'error', got {status!r}")
    return f"::result {status}\n{message}\n::end"
