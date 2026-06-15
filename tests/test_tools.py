import sys

import pytest

from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


@pytest.fixture
def registry(tmp_path):
    return ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)


def test_write_then_read_file(registry, tmp_path):
    status, message = registry.execute(
        Action(verb="write_file", args={"path": "hello.txt"}, body="hi there")
    )
    assert status == "ok"
    assert (tmp_path / "hello.txt").read_text() == "hi there"

    status, message = registry.execute(
        Action(verb="read_file", args={"path": "hello.txt"}, body="")
    )
    assert status == "ok"
    assert "hi there" in message


def test_read_missing_file_is_error(registry):
    status, message = registry.execute(
        Action(verb="read_file", args={"path": "nope.txt"}, body="")
    )
    assert status == "error"
    assert "nope.txt" in message


def test_list_dir(registry, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    status, message = registry.execute(
        Action(verb="list_dir", args={"path": "."}, body="")
    )
    assert status == "ok"
    assert "a.txt" in message and "b.txt" in message


def test_run_command_captures_output(registry):
    status, message = registry.execute(
        Action(
            verb="run_command",
            args={"cmd": f'{sys.executable} -c "print(123)"'},
            body="",
        )
    )
    assert status == "ok"
    assert "123" in message


def test_unknown_verb_is_error(registry):
    status, message = registry.execute(
        Action(verb="fly_to_moon", args={}, body="")
    )
    assert status == "error"
    assert "fly_to_moon" in message


def test_sandbox_escape_is_error(registry):
    status, message = registry.execute(
        Action(verb="read_file", args={"path": "../escape.txt"}, body="")
    )
    assert status == "error"
