"""Single-agent text-protocol loop: emit -> parse one action -> execute ->
feed result back, until 'done', no action, or the step cap."""
from __future__ import annotations

import inspect
from dataclasses import dataclass

from orchestrator.protocol import ProtocolError, parse_action, serialize_result

_FORMAT_REMINDER = (
    "Could not parse an action. Emit exactly one action block:\n"
    "::action <verb>\nkey: value\n---\noptional body\n::end"
)


@dataclass
class AgentResult:
    transcript: list[dict]
    stopped_reason: str


class Agent:
    def __init__(
        self,
        client,
        registry,  # any object with execute(action) -> (status, message)
        model: str,
        system_prompt: str,
        max_steps: int = 50,
        terminal_verbs: set[str] | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.terminal_verbs = terminal_verbs or {"done"}
        # Corrective reminder names THIS agent's terminal verb(s), so the dominant
        # (task_complete) is not wrongly told to emit the worker's "done".
        _verbs = " or ".join(sorted(self.terminal_verbs))
        self._reminder = (
            f"{_FORMAT_REMINDER}\nWhen the task is finished, emit a terminal "
            f"action ({_verbs})."
        )

    async def run(self, task: str) -> AgentResult:
        # NOTE: exceptions from ``client.complete`` (LM Studio unreachable,
        # timeouts) propagate by design — per spec section 7 those are handled
        # at the run-start / orchestrator layer, not inside this loop.
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
                # A malformed reply consumes a step by design: repeated bad
                # output eventually trips the max_steps backstop.
                messages.append(
                    {
                        "role": "user",
                        "content": serialize_result(
                            "error", f"{exc}\n\n{self._reminder}"
                        ),
                    }
                )
                continue

            if action is None:
                reason = "no_action"
                break
            if action.verb in self.terminal_verbs:
                reason = action.verb
                break

            result = self.registry.execute(action)
            if inspect.isawaitable(result):
                result = await result
            status, message = result
            if status == "stop":
                reason = message
                break
            messages.append(
                {"role": "user", "content": serialize_result(status, message)}
            )

        return AgentResult(transcript=messages, stopped_reason=reason)
