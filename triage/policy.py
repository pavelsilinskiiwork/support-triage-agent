"""The rules the model cannot talk its way around. Plain code, no model, easy to test.

Two checks:
- route(): before planning, decide whether a human must handle the ticket whatever the plan says.
- verify(): after planning, check the proposed actions. Violations are either terminal (escalate at once) or fixable
  (the model gets one more try with a fixed description of what was wrong).

Everything the policy trusts comes from the helpdesk metadata (charge age and amount) or from the runbooks, never
from the ticket text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Plan, Ticket
from .prompts import CANARY

AUTO_REFUND_LIMIT_USD = 50.0
REFUND_WINDOW_DAYS = 14
MIN_CONFIDENCE = 0.6
MAX_REPLY_CHARS = 1200

KNOWN_ACTIONS = frozenset(
    {"send_reply", "request_info", "send_password_reset", "issue_refund", "disable_2fa", "change_account_email",
     "escalate_to_human"}
)
PRIVILEGED = frozenset({"issue_refund", "send_password_reset", "disable_2fa", "change_account_email"})
NEVER_AUTO = frozenset({"disable_2fa", "change_account_email"})
REPLY_ACTIONS = frozenset({"send_reply", "request_info"})
QUEUES = frozenset({"tier2", "security", "incident", "privacy", "billing"})
ESCALATE_CATEGORIES = {"security": "security", "outage": "incident", "privacy": "privacy", "other": "tier2"}

DESCRIPTIONS = {
    "invalid_plan": "the reply was not a valid plan in the required JSON format",
    "empty_plan": "the plan had no actions",
    "missing_citation": "the plan did not cite a runbook that covers the actions it uses",
    "unknown_runbook": "the plan cited a runbook id that was not provided",
    "empty_reply": "a reply to the customer was empty",
    "reply_too_long": "a reply to the customer was too long",
    "url_not_in_runbooks": "a reply contained a link that does not appear in the cited runbooks",
    "unknown_action": "the plan used a tool that does not exist",
    "never_auto": "the plan used a tool that only a person may use",
    "bad_refund_amount": "the refund amount was missing or not a positive number",
    "refund_over_limit": "the refund is above the automatic limit",
    "refund_ineligible": "the last charge is not eligible for an automatic refund",
    "refund_exceeds_charge": "the refund is larger than the last charge",
    "injection_privileged": "the ticket contains instruction-like text and the plan uses a privileged tool",
    "canary_leak": "a reply contained internal reference text",
}
FIXABLE = frozenset(
    {"invalid_plan", "empty_plan", "missing_citation", "unknown_runbook", "empty_reply", "reply_too_long",
     "url_not_in_runbooks"}
)


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str = ""

    @property
    def fixable(self) -> bool:
        return self.code in FIXABLE

    @property
    def description(self) -> str:
        return DESCRIPTIONS.get(self.code, self.code)


def route(category: str, urgency: str, confidence: float, error: str | None = None) -> dict | None:
    """A reason to hand the ticket to a person before any plan is made, or None."""
    if error:
        return {"queue": "tier2", "reason": f"classification failed ({error})"}
    if category in ESCALATE_CATEGORIES:
        return {"queue": ESCALATE_CATEGORIES[category], "reason": f"category '{category}' is handled by a person"}
    if urgency == "critical":
        return {"queue": "tier2", "reason": "critical urgency"}
    if confidence < MIN_CONFIDENCE:
        return {"queue": "tier2", "reason": f"low classification confidence ({confidence:.2f})"}
    return None


_URL = re.compile(r"https?://[^\s<>\"')\]]+")


def urls_in(text: str) -> set[str]:
    return {u.rstrip(".,;:!?") for u in _URL.findall(text)}


def _refund_problems(ticket: Ticket, amount_raw) -> list[Violation]:
    try:
        amount = float(amount_raw)
    except (TypeError, ValueError):
        return [Violation("bad_refund_amount")]
    if amount != amount or amount <= 0:  # NaN or non-positive
        return [Violation("bad_refund_amount")]
    out = []
    if amount > AUTO_REFUND_LIMIT_USD:
        out.append(Violation("refund_over_limit", f"{amount:.2f} > {AUTO_REFUND_LIMIT_USD:.0f}"))
    days, charged = ticket.customer.last_charge_days_ago, ticket.customer.last_charge_amount
    if days is None or charged is None or days > REFUND_WINDOW_DAYS:
        out.append(Violation("refund_ineligible", f"last charge age: {days}"))
    elif amount > charged:
        out.append(Violation("refund_exceeds_charge", f"{amount:.2f} > {charged:.2f}"))
    return out


def verify(
    ticket: Ticket,
    plan: Plan | None,
    *,
    available: dict[str, str],
    flags: list[str],
    canary: str = CANARY,
) -> list[Violation]:
    """Check a plan. `available` maps runbook id to text for the runbooks the model was shown."""
    if plan is None:
        return [Violation("invalid_plan")]
    if not plan.actions:
        return [Violation("empty_plan")]

    out: list[Violation] = []
    names = [a.name for a in plan.actions]
    cited = list(plan.cited_runbooks)

    for rid in cited:
        if rid not in available:
            out.append(Violation("unknown_runbook", rid[:60]))
    cited_known = [rid for rid in cited if rid in available]
    if not cited_known and "escalate_to_human" not in names:
        out.append(Violation("missing_citation"))
    allowed_urls: set[str] = set()
    for rid in cited_known:
        allowed_urls |= urls_in(available[rid])

    if flags and any(n in PRIVILEGED for n in names):
        out.append(Violation("injection_privileged", ",".join(flags)))

    for action in plan.actions:
        if action.name not in KNOWN_ACTIONS:
            out.append(Violation("unknown_action", action.name[:40]))
        elif action.name in NEVER_AUTO:
            out.append(Violation("never_auto", action.name))
        elif action.name == "issue_refund":
            out.extend(_refund_problems(ticket, action.params.get("amount_usd")))
            if "billing-refund-policy" not in cited_known:
                out.append(Violation("missing_citation", "refund needs billing-refund-policy"))
        elif action.name == "send_password_reset" and "account-password-reset" not in cited_known:
            out.append(Violation("missing_citation", "reset needs account-password-reset"))
        elif action.name in REPLY_ACTIONS:
            text = action.params.get("text")
            if not isinstance(text, str) or not text.strip():
                out.append(Violation("empty_reply"))
                continue
            if len(text) > MAX_REPLY_CHARS:
                out.append(Violation("reply_too_long", str(len(text))))
            if canary and canary in text:
                out.append(Violation("canary_leak"))
            if any(u not in allowed_urls for u in urls_in(text)):
                out.append(Violation("url_not_in_runbooks"))
    # one violation per code keeps feedback and logs short
    seen, unique = set(), []
    for v in out:
        if v.code not in seen:
            seen.add(v.code)
            unique.append(v)
    return unique


def is_terminal(violations: list[Violation]) -> bool:
    return any(not v.fixable for v in violations)
