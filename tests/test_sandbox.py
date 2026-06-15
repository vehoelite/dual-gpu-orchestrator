import pytest

from orchestrator.sandbox import Sandbox, SandboxError


def test_resolve_simple_path(tmp_path):
    sandbox = Sandbox(tmp_path)
    resolved = sandbox.resolve("notes.md")
    assert resolved == (tmp_path / "notes.md").resolve()


def test_resolve_nested_path(tmp_path):
    sandbox = Sandbox(tmp_path)
    resolved = sandbox.resolve("sub/dir/file.txt")
    assert resolved == (tmp_path / "sub" / "dir" / "file.txt").resolve()


def test_resolve_dot_is_root(tmp_path):
    sandbox = Sandbox(tmp_path)
    assert sandbox.resolve(".") == tmp_path.resolve()


def test_escape_with_parent_raises(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.resolve("../secret.txt")


def test_absolute_path_outside_root_raises(tmp_path):
    sandbox = Sandbox(tmp_path)
    with pytest.raises(SandboxError):
        sandbox.resolve(str(tmp_path.parent / "elsewhere.txt"))
