import pytest

from triage import policy
from triage.models import Action, Plan
from triage.prompts import CANARY

REFUND_TEXT = "Refunds ... https://northwind.example/legal/refunds"
AVAILABLE = {
    "billing-refund-policy": REFUND_TEXT,
    "account-password-reset": "Use the tool.",
    "bug-report-intake": "Ask for details. https://status.northwind.example/known-issues",
}


def make_plan(*actions, cited=("billing-refund-policy",)):
    return Plan(actions=[Action(name=n, params=p) for n, p in actions], cited_runbooks=list(cited))


def codes(ticket, p, flags=()):
    return [v.code for v in policy.verify(ticket, p, available=AVAILABLE, flags=list(flags))]


def test_clean_refund_passes(make_ticket):
    assert codes(make_ticket(days=1, amount=29), make_plan(("issue_refund", {"amount_usd": 29}))) == []


@pytest.mark.parametrize("days,amount,asked,expected", [
    (1, 480, 480, ["refund_over_limit"]),
    (30, 29, 29, ["refund_ineligible"]),
    (None, None, 10, ["refund_ineligible"]),
    (1, 20, 25, ["refund_exceeds_charge"]),
    (60, 400, 300, ["refund_over_limit", "refund_ineligible"]),
])
def test_refund_rules_use_trusted_metadata(make_ticket, days, amount, asked, expected):
    assert codes(make_ticket(days=days, amount=amount), make_plan(("issue_refund", {"amount_usd": asked}))) == expected


@pytest.mark.parametrize("bad", [None, "abc", -5, 0, float("nan")])
def test_bad_refund_amounts(make_ticket, bad):
    assert codes(make_ticket(), make_plan(("issue_refund", {"amount_usd": bad}))) == ["bad_refund_amount"]


def test_refund_needs_the_refund_runbook(make_ticket):
    p = make_plan(("issue_refund", {"amount_usd": 10}), cited=("bug-report-intake",))
    assert "missing_citation" in codes(make_ticket(), p)


def test_never_auto_and_unknown_tools(make_ticket):
    assert codes(make_ticket(), make_plan(("disable_2fa", {}))) == ["never_auto"]
    result = codes(make_ticket(), make_plan(("disable_2fa", {}), ("change_account_email", {"new_email": "x@y.z"})))
    assert "never_auto" in result
    assert codes(make_ticket(), make_plan(("delete_everything", {}))) == ["unknown_action"]


def test_privileged_action_with_injection_flags_is_terminal(make_ticket):
    v = policy.verify(make_ticket(), make_plan(("issue_refund", {"amount_usd": 10})),
                      available=AVAILABLE, flags=["ignore_instructions"])
    assert [x.code for x in v] == ["injection_privileged"]
    assert policy.is_terminal(v)


def test_flags_alone_do_not_block_an_unprivileged_plan(make_ticket):
    p = make_plan(("send_reply", {"text": "See https://status.northwind.example/known-issues"}), cited=("bug-report-intake",))
    assert codes(make_ticket(), p, flags=["tool_names"]) == []


def test_citations(make_ticket):
    p = make_plan(("send_reply", {"text": "hi"}), cited=())
    assert codes(make_ticket(), p) == ["missing_citation"]
    p = make_plan(("send_reply", {"text": "hi"}), cited=("made-up",))
    assert set(codes(make_ticket(), p)) == {"unknown_runbook", "missing_citation"}


def test_reply_checks(make_ticket):
    t = make_ticket()
    assert codes(t, make_plan(("send_reply", {"text": "  "}))) == ["empty_reply"]
    assert codes(t, make_plan(("send_reply", {"text": "x" * 1300}))) == ["reply_too_long"]
    assert codes(t, make_plan(("send_reply", {"text": f"ref {CANARY}"}))) == ["canary_leak"]
    assert codes(t, make_plan(("send_reply", {"text": "go to http://evil.example/x"}))) == ["url_not_in_runbooks"]
    ok = "Policy: https://northwind.example/legal/refunds."
    assert codes(t, make_plan(("send_reply", {"text": ok}))) == []


def test_plan_shape(make_ticket):
    assert codes(make_ticket(), None) == ["invalid_plan"]
    assert codes(make_ticket(), Plan()) == ["empty_plan"]


def test_fixable_versus_terminal(make_ticket):
    v = policy.verify(make_ticket(), make_plan(("send_reply", {"text": ""})), available=AVAILABLE, flags=[])
    assert not policy.is_terminal(v)
    v = policy.verify(make_ticket(), make_plan(("disable_2fa", {})), available=AVAILABLE, flags=[])
    assert policy.is_terminal(v)


@pytest.mark.parametrize("category,urgency,conf,error,queue", [
    ("security", "normal", 0.9, None, "security"),
    ("outage", "normal", 0.9, None, "incident"),
    ("privacy", "normal", 0.9, None, "privacy"),
    ("other", "normal", 0.9, None, "tier2"),
    ("billing", "critical", 0.9, None, "tier2"),
    ("billing", "normal", 0.3, None, "tier2"),
    ("billing", "normal", 0.9, "llm_error", "tier2"),
])
def test_route_escalates(category, urgency, conf, error, queue):
    assert policy.route(category, urgency, conf, error)["queue"] == queue


def test_route_lets_ordinary_tickets_through():
    assert policy.route("billing", "normal", 0.9) is None
    assert policy.route("how_to", "low", 0.6) is None
