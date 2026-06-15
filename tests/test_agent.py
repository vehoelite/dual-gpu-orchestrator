from orchestrator.agent import Agent, AgentResult
from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


class FakeClient:
    """Returns scripted completions in order, ignoring the prompt."""

    def __init__(self, scripted: list[str]) -> None:
        self._scripted = list(scripted)
        self.calls: list[list[dict]] = []

    async def complete(self, model, messages, temperature=0.7):
        self.calls.append(messages)
        return self._scripted.pop(0)


def _agent(tmp_path, scripted, max_steps=10):
    registry = ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)
    return Agent(
        client=FakeClient(scripted),
        registry=registry,
        model="test-model",
        system_prompt="You are a worker.",
        max_steps=max_steps,
    )


async def test_writes_file_then_done(tmp_path):
    scripted = [
        "I'll create the file.\n::action write_file\npath: out.txt\n---\nhi\n::end",
        "All set.\n::action done\n::end",
    ]
    agent = _agent(tmp_path, scripted)
    result = await agent.run("create out.txt with hi")
    assert isinstance(result, AgentResult)
    assert result.stopped_reason == "done"
    assert (tmp_path / "out.txt").read_text() == "hi"


async def test_result_is_fed_back_into_conversation(tmp_path):
    scripted = [
        "::action write_file\npath: a.txt\n---\nx\n::end",
        "::action done\n::end",
    ]
    agent = _agent(tmp_path, scripted)
    await agent.run("task")
    # Second model call must include the ::result ok from the first action.
    second_call_messages = agent.client.calls[1]
    assert any("::result ok" in m["content"] for m in second_call_messages)


async def test_no_action_stops(tmp_path):
    agent = _agent(tmp_path, ["I have nothing to do."])
    result = await agent.run("task")
    assert result.stopped_reason == "no_action"


async def test_malformed_action_gets_corrective_reprompt(tmp_path):
    scripted = [
        "::action write_file\npath: a.txt\n",  # missing ::end -> ProtocolError
        "::action done\n::end",
    ]
    agent = _agent(tmp_path, scripted)
    result = await agent.run("task")
    assert result.stopped_reason == "done"
    second_call_messages = agent.client.calls[1]
    assert any("::result error" in m["content"] for m in second_call_messages)


async def test_max_steps_stops(tmp_path):
    # Always emits a valid action, never 'done'.
    looping = "::action list_dir\npath: .\n::end"
    agent = _agent(tmp_path, [looping] * 5, max_steps=3)
    result = await agent.run("task")
    assert result.stopped_reason == "max_steps"
    assert len(agent.client.calls) == 3


async def test_repeated_malformed_hits_max_steps(tmp_path):
    # A model that always emits malformed blocks must trip the backstop, not
    # raise — each bad reply consumes a step.
    malformed = "::action write_file\npath: a.txt\n"  # missing ::end
    agent = _agent(tmp_path, [malformed] * 5, max_steps=3)
    result = await agent.run("task")
    assert result.stopped_reason == "max_steps"
    assert len(agent.client.calls) == 3
    assert any(
        "::result error" in m["content"]
        for m in result.transcript
        if m["role"] == "user"
    )
