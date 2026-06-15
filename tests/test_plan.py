import pytest

from orchestrator.plan import Plan, PlanError, Step, parse_checklist


def test_from_descriptions_all_pending():
    plan = Plan.from_descriptions(["a", "b"])
    assert [s.status for s in plan.steps] == ["pending", "pending"]
    assert plan.steps[0].description == "a"


def test_mark_in_progress_and_done():
    plan = Plan.from_descriptions(["a", "b"])
    plan.mark_in_progress(0)
    assert plan.steps[0].status == "in_progress"
    plan.mark_done(0)
    assert plan.steps[0].status == "done"


def test_bad_index_raises():
    plan = Plan.from_descriptions(["a"])
    with pytest.raises(PlanError):
        plan.mark_done(5)


def test_revise_replaces_steps():
    plan = Plan.from_descriptions(["a", "b"])
    plan.mark_done(0)
    plan.revise(["x", "y", "z"])
    assert [s.description for s in plan.steps] == ["x", "y", "z"]
    assert all(s.status == "pending" for s in plan.steps)


def test_all_done():
    plan = Plan.from_descriptions(["a", "b"])
    assert not plan.all_done()
    plan.mark_done(0)
    plan.mark_done(1)
    assert plan.all_done()


def test_empty_plan_not_all_done():
    assert not Plan().all_done()


def test_signature_changes_with_status():
    plan = Plan.from_descriptions(["a"])
    sig1 = plan.signature()
    plan.mark_done(0)
    assert plan.signature() != sig1


def test_render_contains_status_and_index():
    plan = Plan.from_descriptions(["write code"])
    plan.mark_done(0)
    out = plan.render()
    assert "1/1 done" in out
    assert "[done] 0. write code" in out


def test_parse_checklist_numbered():
    assert parse_checklist("1. do a\n2. do b\n3. do c") == ["do a", "do b", "do c"]


def test_parse_checklist_bullets_and_noise():
    text = "Here is the plan:\n- alpha\n* beta\n\nThanks!"
    assert parse_checklist(text) == ["alpha", "beta"]


def test_parse_checklist_paren_numbers():
    assert parse_checklist("1) first\n2) second") == ["first", "second"]


def test_parse_checklist_empty():
    assert parse_checklist("no list here") == []
