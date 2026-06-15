"""Single-agent text-protocol loop: emit -> parse one action -> execute ->
feed result back, until 'done', no action, or the step cap."""
from __future__ import annotations

from dataclasses import dataclass

from orchestrator.protocol import ProtocolError, parse_action, serialize_result
from orchestrator.tools import ToolRegistry

_FORMAT_REMINDER = (
    "Could not parse an action. Emit exactly one action block:\n"
    "::action <verb>\nkey: value\n---\noptional body\n::end\n"
    "When the task is finished, emit ::action done\\n::end."
)


@dataclass
class AgentResult:
    transcript: list[dict]
    stopped_reason: str


class Agent:
    def __init__(
        self,
        client,
        registry: ToolRegistry,
        model: str,
        system_prompt: str,
        max_steps: int = 50,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    async def run(self, task: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        reason = "max_steps"
        for _ in range(self.max_steps):
            reply = await self.client.complete(model=self.model, messages=messages)
            messages.append({"role": "assistant", "content": reply})

            try:
                action = parse_action(reply)
            except ProtocolError as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": serialize_result(
                            "error", f"{exc}\n\n{_FORMAT_REMINDER}"
                        ),
                    }
                )
                continue

            if action is None:
                reason = "no_action"
                break
            if action.verb == "done":
                reason = "done"
                break

            status, message = self.registry.execute(action)
            messages.append(
                {"role": "user", "content": serialize_result(status, message)}
            )

        return AgentResult(transcript=messages, stopped_reason=reason)
