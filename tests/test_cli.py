import pytest

from orchestrator.cli import _select_models
from orchestrator.orchestrator import WORKER_PROMPT


def test_select_defaults_to_load_order():
    assert _select_models(["a", "b"]) == ("a", "b")


def test_select_single_model_used_for_both():
    assert _select_models(["only"]) == ("only", "only")


def test_select_by_substring():
    models = ["qwen3.5-4b-worker", "omnicoder-9b-dominant"]
    assert _select_models(models, dominant="9b", worker="4b") == (
        "omnicoder-9b-dominant",
        "qwen3.5-4b-worker",
    )


def test_select_unmatched_hint_raises():
    with pytest.raises(ValueError):
        _select_models(["a", "b"], dominant="nope")


def test_worker_prompt_has_concrete_example():
    # Weak models parrot placeholders; the prompt must show a filled-in example.
    assert "path: hello.py" in WORKER_PROMPT
    assert "<verb>" not in WORKER_PROMPT


def test_research_hint_has_concrete_example():
    from orchestrator.orchestrator import RESEARCH_HINT
    assert "::action research" in RESEARCH_HINT
    assert "query:" in RESEARCH_HINT
