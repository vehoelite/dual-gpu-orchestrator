"""Parse the structured text action protocol and serialize results.

Deliberately forgiving: the worker role is filled by MANY different (often weak,
often reasoning) local models, so the parser tolerates the common ways they
deviate rather than rejecting and looping:
  - a missing ``::end`` (block closes at end-of-text)
  - a stray/empty ``---`` separator (no body)
  - ``key=value`` as well as ``key: value``
  - inline args on the ``::action`` line
  - surrounding quotes on a value (command quoting is preserved)
  - reasoning/preamble before the action — we anchor on the LAST ``::action``,
    so a model that thinks first (and may even mention the format) still parses.

The only hard failure is an ``::action`` with no verb at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_MARKER = "::action"
_END = "::end"
# A real action marker begins a line (optionally indented). This ignores a bare
# "::action" mentioned mid-sentence in prose/reasoning.
_MARKER_RE = re.compile(rf"(?:^|\n)[ \t]*{re.escape(_MARKER)}")


class ProtocolError(Exception):
    """Raised when an ``::action`` marker is present but has no verb."""


@dataclass(frozen=True)
class Action:
    verb: str
    args: dict[str, str]
    body: str


def parse_action(text: str) -> Action | None:
    """Return the Action in ``text``.

    ``None`` if there is no ``::action`` marker at all (the model is just
    talking). Raises ``ProtocolError`` only if a marker is present with no verb.
    Everything else is parsed leniently (see module docstring)."""
    text = text.replace("\r\n", "\n")
    # Anchor on the LAST line-starting marker: reasoning models think first then
    # act, so the real block is last; a bare "::action" mid-prose is ignored.
    matches = list(_MARKER_RE.finditer(text))
    if not matches:
        return None

    after = text[matches[-1].end():]
    newline = after.find("\n")
    if newline == -1:  # single-line block: "::action verb key: value"
        header, rest = after, ""
    else:
        header, rest = after[:newline], after[newline + 1:]

    end = rest.find(_END)
    block = rest if end == -1 else rest[:end]  # missing ::end -> close at EOT

    verb, inline = _split_verb(header)
    if not verb:
        raise ProtocolError("action is missing a verb")

    args, body = _parse_args_and_body(inline, block)
    return Action(verb=verb, args=args, body=body)


def _split_verb(header: str) -> tuple[str, str]:
    """('write_file', 'path: x') from a header line; inline part may be empty."""
    parts = header.strip().split(None, 1)
    if not parts:
        return "", ""
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _parse_args_and_body(inline: str, block: str) -> tuple[dict[str, str], str]:
    args: dict[str, str] = {}
    if inline:
        key, value = _split_pair(inline)
        if key is not None:
            args[key] = value

    body_lines: list[str] | None = None
    lines = block.split("\n")
    for i, line in enumerate(lines):
        if body_lines is not None:
            body_lines.append(line)
            continue
        stripped = line.strip()
        if stripped == "---":  # separator -> the rest is body
            body_lines = []
            continue
        if stripped == "":
            continue
        key, value = _split_pair(line)
        if key is None:  # not a key:value/key=value line -> body starts here
            body_lines = lines[i:]
            break
        args[key] = value

    body = "\n".join(body_lines).strip("\n") if body_lines is not None else ""
    return args, body


def _split_pair(line: str) -> tuple[str | None, str]:
    """Parse 'key: value' or 'key=value'. Returns (None, '') for a non-pair
    line (e.g. prose), so it can be treated as body. A key may not contain
    spaces — that rules out prose that merely contains a colon."""
    for sep in (":", "="):
        if sep in line:
            key, _, value = line.partition(sep)
            key = key.strip()
            if key and " " not in key:
                return key, _strip_quotes(value.strip())
    return None, ""


def _strip_quotes(value: str) -> str:
    """Strip a single matching pair of surrounding quotes (so path: "x" -> x),
    but leave inner quotes alone (so a command's quoting survives)."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def serialize_result(status: str, message: str) -> str:
    """Render an engine -> model result block."""
    if status not in ("ok", "error"):
        raise ValueError(f"status must be 'ok' or 'error', got {status!r}")
    return f"::result {status}\n{message}\n::end"
