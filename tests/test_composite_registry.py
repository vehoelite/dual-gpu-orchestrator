from orchestrator.composite_registry import CompositeRegistry
from orchestrator.protocol import Action
from orchestrator.sandbox import Sandbox
from orchestrator.tools import ToolRegistry


class FakeResearcher:
    def __init__(self, answer="research answer", boom=False):
        self.answer = answer
        self.boom = boom
        self.received = None

    async def research(self, query):
        self.received = query
        if self.boom:
            raise RuntimeError("network down")
        return self.answer


def _composite(tmp_path, researcher):
    return CompositeRegistry(
        tool_registry=ToolRegistry(Sandbox(tmp_path), command_timeout=10.0),
        researcher=researcher,
    )


async def test_research_routes_to_researcher(tmp_path):
    r = FakeResearcher(answer="found it")
    comp = _composite(tmp_path, r)
    status, msg = await comp.execute(Action("research", {"query": "news?"}, ""))
    assert status == "ok"
    assert msg == "found it"
    assert r.received == "news?"


async def test_research_uses_body_when_no_query_arg(tmp_path):
    r = FakeResearcher()
    comp = _composite(tmp_path, r)
    await comp.execute(Action("research", {}, "  body query  "))
    assert r.received == "body query"


async def test_research_missing_query_is_error(tmp_path):
    comp = _composite(tmp_path, FakeResearcher())
    status, msg = await comp.execute(Action("research", {}, "   "))
    assert status == "error"


async def test_research_exception_is_error(tmp_path):
    comp = _composite(tmp_path, FakeResearcher(boom=True))
    status, msg = await comp.execute(Action("research", {"query": "x"}, ""))
    assert status == "error"
    assert "network down" in msg


async def test_file_verb_routes_to_tool_registry(tmp_path):
    comp = _composite(tmp_path, FakeResearcher())
    status, msg = await comp.execute(
        Action("write_file", {"path": "a.txt"}, "hi")
    )
    assert status == "ok"
    assert (tmp_path / "a.txt").read_text() == "hi"
