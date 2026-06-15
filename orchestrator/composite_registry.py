"""Routes the `research` verb to an McpResearcher (async, network) and every
other verb to the first-party ToolRegistry (sync). The Phase 2 agent loop is
await-tolerant, so this async execute() drops in for the worker."""
from __future__ import annotations

from orchestrator.protocol import Action


class CompositeRegistry:
    def __init__(self, tool_registry, researcher) -> None:
        self.tool_registry = tool_registry
        self.researcher = researcher

    async def execute(self, action: Action) -> tuple[str, str]:
        if action.verb == "research":
            query = action.args.get("query") or action.body.strip()
            if not query:
                return "error", "research needs a 'query' arg or a body"
            try:
                answer = await self.researcher.research(query)
            except Exception as exc:  # surface any failure to the agent, don't crash
                return "error", f"research failed: {exc}"
            return "ok", answer
        return self.tool_registry.execute(action)
