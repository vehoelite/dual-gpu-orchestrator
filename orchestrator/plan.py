"""The dominant's checklist: ordered steps with status, plus transitions.

Also hosts ``parse_checklist`` since turning checklist text into steps is part
of building a Plan (used by the planner and the coordination verbs)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


class PlanError(Exception):
    """Raised on an invalid plan operation (e.g. a bad step index)."""


@dataclass
class Step:
    description: str
    status: str = "pending"  # "pending" | "in_progress" | "done"


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def from_descriptions(cls, descriptions: list[str]) -> "Plan":
        return cls(steps=[Step(d) for d in descriptions])

    def _check(self, index: int) -> None:
        if not 0 <= index < len(self.steps):
            raise PlanError(f"no step {index} (plan has {len(self.steps)} steps)")

    def mark_in_progress(self, index: int) -> None:
        self._check(index)
        self.steps[index].status = "in_progress"

    def mark_done(self, index: int) -> None:
        self._check(index)
        self.steps[index].status = "done"

    def revise(self, descriptions: list[str]) -> None:
        self.steps = [Step(d) for d in descriptions]

    def all_done(self) -> bool:
        return len(self.steps) > 0 and all(s.status == "done" for s in self.steps)

    def signature(self) -> tuple:
        return tuple((s.description, s.status) for s in self.steps)

    def render(self) -> str:
        done = sum(1 for s in self.steps if s.status == "done")
        lines = [f"Plan ({done}/{len(self.steps)} done):"]
        for i, s in enumerate(self.steps):
            lines.append(f"[{s.status}] {i}. {s.description}")
        return "\n".join(lines)


def parse_checklist(text: str) -> list[str]:
    """Extract step descriptions from a numbered/bulleted checklist."""
    steps: list[str] = []
    for line in text.splitlines():
        match = _LINE_RE.match(line)
        if match:
            steps.append(match.group(1).strip())
    return steps
