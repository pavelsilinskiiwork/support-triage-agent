import pytest

from triage.helpdesk import ESCALATION_REPLY, Helpdesk, ToolError, escalate
from triage.llm import LLMError
from triage.models import Action
from triage.variants import SingleAgent, make_agent

from .helpers import ScriptedLLM, refund, reply, single


def names(result):
    return [e["name"] for e in result.executed]


def test_naive_single_executes_whatever_the_model_says(runbooks, make_ticket):
    llm = ScriptedLLM(single("billing", "auto_resolve", reply(), refund(500)))
    result = SingleAgent(llm, runbooks).run(make_ticket(days=90, amount=29))
    assert "issue_refund" in names(result)
    assert result.executed[-1]["params"]["amount_usd"] == 500
    assert result.llm_calls == 1


def test_naive_single_reports_tool_errors(runbooks, make_ticket):
    llm = ScriptedLLM(single("billing", "auto_resolve", ("delete_everything", {}), reply()))
    result = SingleAgent(llm, runbooks).run(make_ticket())
    assert names(result) == ["send_reply"] and "unknown tool" in result.reasons[0]


def test_guarded_single_applies_the_same_policy(runbooks, make_ticket):
    llm = ScriptedLLM(single("billing", "auto_resolve", reply(), refund(500)))
    result = SingleAgent(llm, runbooks, guarded=True).run(make_ticket(days=90, amount=29))
    assert result.disposition == "escalate" and "issue_refund" not in names(result)
    assert any("refund_over_limit" in r for r in result.reasons)
    assert result.executed[0]["params"]["text"] == ESCALATION_REPLY


def test_guarded_single_lets_a_clean_plan_through(runbooks, make_ticket):
    llm = ScriptedLLM(single("billing", "auto_resolve", reply(), refund(29)))
    result = SingleAgent(llm, runbooks, guarded=True).run(make_ticket(days=1, amount=29))
    assert names(result) == ["send_reply", "issue_refund"] and result.disposition == "auto_resolve"


def test_guarded_single_escalates_sensitive_categories(runbooks, make_ticket):
    llm = ScriptedLLM(single("security", "auto_resolve", reply()))
    result = SingleAgent(llm, runbooks, guarded=True).run(make_ticket())
    assert result.disposition == "escalate" and result.executed[-1]["params"]["queue"] == "security"


def test_failures_differ_between_naive_and_guarded(runbooks, make_ticket):
    naive = SingleAgent(ScriptedLLM("junk", "junk"), runbooks).run(make_ticket())
    guarded = SingleAgent(ScriptedLLM("junk", "junk"), runbooks, guarded=True).run(make_ticket())
    assert naive.disposition == "no_action" and guarded.disposition == "escalate"
    assert SingleAgent(ScriptedLLM(LLMError("down")), runbooks).run(make_ticket()).disposition == "no_action"
    assert SingleAgent(ScriptedLLM(LLMError("down")), runbooks, guarded=True).run(make_ticket()).disposition == "escalate"


def test_make_agent(runbooks):
    llm = ScriptedLLM()
    assert make_agent("single", llm, runbooks).variant == "single"
    assert make_agent("single_guarded", llm, runbooks).variant == "single_guarded"
    assert make_agent("multi", llm, runbooks).variant == "multi"
    with pytest.raises(ValueError):
        make_agent("other", llm, runbooks)


def test_helpdesk_records_and_validates(make_ticket):
    hd, t = Helpdesk(), make_ticket()
    hd.execute(t, Action(name="issue_refund", params={"amount_usd": "12.5"}))
    assert hd.log[-1]["params"]["amount_usd"] == 12.5 and hd.log[-1]["ticket"] == "T-1"
    for bad in (Action(name="issue_refund", params={}), Action(name="nope")):
        with pytest.raises(ToolError):
            hd.execute(t, bad)
    assert [e["name"] for e in escalate(hd, t, "tier2", "why")] == ["send_reply", "escalate_to_human"]
