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
        "::action\n::end",  # no verb -> ProtocolError (the one hard failure)
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
    malformed = "::action\n::end"  # no verb -> the one hard parse failure
    agent = _agent(tmp_path, [malformed] * 5, max_steps=3)
    result = await agent.run("task")
    assert result.stopped_reason == "max_steps"
    assert len(agent.client.calls) == 3
    assert any(
        "::result error" in m["content"]
        for m in result.transcript
        if m["role"] == "user"
    )


class AsyncEchoRegistry:
    """Async registry: records actions, returns ok; 'halt' returns a stop."""

    def __init__(self):
        self.executed = []

    async def execute(self, action):
        self.executed.append(action)
        if action.verb == "halt":
            return ("stop", "no_progress")
        return ("ok", f"did {action.verb}")


async def test_terminal_verb_configurable(tmp_path):
    reg = AsyncEchoRegistry()
    client = FakeClient(["::action task_complete\n::end"])
    agent = Agent(
        client=client, registry=reg, model="m", system_prompt="s",
        max_steps=5, terminal_verbs={"task_complete"},
    )
    result = await agent.run("go")
    assert result.stopped_reason == "task_complete"
    assert reg.executed == []  # terminal verb is not executed by the registry


async def test_async_registry_is_awaited(tmp_path):
    reg = AsyncEchoRegistry()
    client = FakeClient(["::action foo\n::end", "::action done\n::end"])
    agent = Agent(client=client, registry=reg, model="m", system_prompt="s", max_steps=5)
    result = await agent.run("go")
    assert result.stopped_reason == "done"
    assert [a.verb for a in reg.executed] == ["foo"]
    assert any("::result ok" in m["content"] for m in agent.client.calls[1])


async def test_stop_status_ends_run(tmp_path):
    reg = AsyncEchoRegistry()
    client = FakeClient(["::action halt\n::end", "::action done\n::end"])
    agent = Agent(client=client, registry=reg, model="m", system_prompt="s", max_steps=5)
    result = await agent.run("go")
    assert result.stopped_reason == "no_progress"
    assert len(agent.client.calls) == 1


async def test_corrective_reminder_names_terminal_verb(tmp_path):
    # A malformed reply from a dominant-style agent should be told to emit its
    # own terminal verb (task_complete), not the worker's "done".
    reg = AsyncEchoRegistry()
    client = FakeClient([
        "::action write_file\npath: a.txt\n",  # malformed (no ::end)
        "::action task_complete\n::end",
    ])
    agent = Agent(
        client=client, registry=reg, model="m", system_prompt="s",
        max_steps=5, terminal_verbs={"task_complete"},
    )
    result = await agent.run("go")
    assert result.stopped_reason == "task_complete"
    reminder = agent.client.calls[1][-1]["content"]
    assert "task_complete" in reminder
    assert "done" not in reminder


class RecordingSink:
    def __init__(self):
        self.events = []

    def emit(self, event):
        self.events.append(event)


async def test_agent_emits_event_sequence(tmp_path):
    sink = RecordingSink()
    registry = ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)
    agent = Agent(
        client=FakeClient([
            "::action write_file\npath: a.txt\n---\nhi\n::end",
            "::action done\n::end",
        ]),
        registry=registry, model="m", system_prompt="s", max_steps=5,
        sink=sink, agent_label="worker",
    )
    await agent.run("t")
    assert [e["type"] for e in sink.events] == [
        "message", "action", "result", "message", "action",
    ]
    assert sink.events[1]["verb"] == "write_file"
    assert sink.events[1]["agent"] == "worker"
    assert sink.events[2]["status"] == "ok"


async def test_agent_emits_parse_error(tmp_path):
    sink = RecordingSink()
    registry = ToolRegistry(sandbox=Sandbox(tmp_path), command_timeout=10.0)
    agent = Agent(
        client=FakeClient(["::action\n::end", "::action done\n::end"]),
        registry=registry, model="m", system_prompt="s", max_steps=5, sink=sink,
    )
    await agent.run("t")
    assert any(e["type"] == "parse_error" for e in sink.events)


async def test_resume_continues_from_prior_transcript(tmp_path):
    # A worker "finished" with done; resume tells it to fix its work. It must
    # continue from the SAME conversation (its prior attempt stays in context).
    prior = [
        {"role": "system", "content": "You are a worker."},
        {"role": "user", "content": "create out.txt"},
        {"role": "assistant", "content": "::action done\n::end"},
    ]
    scripted = [
        "::action write_file\npath: out.txt\n---\nfixed\n::end",
        "::action done\n::end",
    ]
    agent = _agent(tmp_path, scripted)
    result = await agent.resume(prior, "::result error\nDo better\n::end")
    assert result.stopped_reason == "done"
    assert (tmp_path / "out.txt").read_text() == "fixed"
    # First model call carries the prior history plus the appended followup turn.
    first_call = agent.client.calls[0]
    assert first_call[:3] == prior
    assert {"role": "user", "content": "::result error\nDo better\n::end"} in first_call


async def test_resume_does_not_mutate_caller_transcript(tmp_path):
    prior = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "::action done\n::end"},
    ]
    agent = _agent(tmp_path, ["::action done\n::end"])
    await agent.resume(prior, "::result error\nredo\n::end")
    assert len(prior) == 3  # caller's list untouched
