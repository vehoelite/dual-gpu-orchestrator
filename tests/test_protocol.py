import pytest

from orchestrator.protocol import (
    Action,
    ProtocolError,
    parse_action,
    serialize_result,
)


def test_no_action_marker_returns_none():
    assert parse_action("I am just thinking out loud.") is None


def test_parse_action_with_args_only():
    text = (
        "Let me read the file.\n"
        "::action read_file\n"
        "path: notes.md\n"
        "::end\n"
    )
    action = parse_action(text)
    assert action == Action(verb="read_file", args={"path": "notes.md"}, body="")


def test_parse_action_with_body():
    text = (
        "::action write_file\n"
        "path: hello.txt\n"
        "---\n"
        "Hello\n"
        "world\n"
        "::end\n"
    )
    action = parse_action(text)
    assert action.verb == "write_file"
    assert action.args == {"path": "hello.txt"}
    assert action.body == "Hello\nworld"


def test_parse_tolerates_surrounding_whitespace():
    text = "  ::action   list_dir  \n   path:   .  \n  ::end  "
    action = parse_action(text)
    assert action.verb == "list_dir"
    assert action.args == {"path": "."}
    assert action.body == ""


def test_missing_end_raises_protocol_error():
    text = "::action read_file\npath: notes.md\n"
    with pytest.raises(ProtocolError):
        parse_action(text)


def test_missing_verb_raises_protocol_error():
    text = "::action\npath: notes.md\n::end\n"
    with pytest.raises(ProtocolError):
        parse_action(text)


def test_serialize_result_ok():
    assert serialize_result("ok", "wrote 5 bytes") == (
        "::result ok\nwrote 5 bytes\n::end"
    )


def test_serialize_result_error():
    assert serialize_result("error", "no such file") == (
        "::result error\nno such file\n::end"
    )
