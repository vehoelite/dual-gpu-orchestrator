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


def test_parse_action_value_with_colon():
    text = "::action fetch\nurl: http://example.com:8080/path\n::end\n"
    action = parse_action(text)
    assert action.args == {"url": "http://example.com:8080/path"}


def test_parse_normalizes_crlf_body():
    text = "::action write_file\r\npath: x.txt\r\n---\r\nHello\r\nworld\r\n::end\r\n"
    action = parse_action(text)
    assert action.args == {"path": "x.txt"}
    assert action.body == "Hello\nworld"


def test_missing_end_is_tolerated():
    # Weak models often omit ::end; close the block at end-of-text.
    text = "::action read_file\npath: notes.md\n"
    action = parse_action(text)
    assert action == Action(verb="read_file", args={"path": "notes.md"}, body="")


def test_stray_separator_and_missing_end():
    # The exact run_command loop we observed: trailing --- and no ::end.
    text = '::action run_command\ncmd: echo "hi" > f.txt\n---'
    action = parse_action(text)
    assert action.verb == "run_command"
    assert action.args == {"cmd": 'echo "hi" > f.txt'}  # inner quotes preserved
    assert action.body == ""


def test_key_equals_value():
    text = "::action write_file\npath=out.txt\n::end"
    action = parse_action(text)
    assert action.args == {"path": "out.txt"}


def test_inline_args_on_action_line():
    text = "::action read_file path: notes.md\n::end"
    action = parse_action(text)
    assert action.verb == "read_file"
    assert action.args == {"path": "notes.md"}


def test_surrounding_quotes_stripped():
    text = '::action write_file\npath: "hello.txt"\n::end'
    action = parse_action(text)
    assert action.args == {"path": "hello.txt"}


def test_reasoning_preamble_anchors_on_last_action():
    # A reasoning model thinks first (and may mention the format), then acts.
    text = (
        "<thinking>I should use the ::action format to write the file.</thinking>\n"
        "::action write_file\n"
        "path: out.txt\n"
        "---\n"
        "hello\n"
        "::end"
    )
    action = parse_action(text)
    assert action.verb == "write_file"
    assert action.args == {"path": "out.txt"}
    assert action.body == "hello"


def test_trailing_prose_does_not_clobber_real_action():
    # Model emits a valid action, then explains it using the keyword in prose.
    # Only line-starting markers count, so the real action wins.
    text = (
        "::action run_command\n"
        "cmd: ls\n"
        "::end\n"
        "I invoked ::action to execute the listing.\n"
    )
    action = parse_action(text)
    assert action.verb == "run_command"
    assert action.args == {"cmd": "ls"}


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
