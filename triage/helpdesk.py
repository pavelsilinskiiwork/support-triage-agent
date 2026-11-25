"""A fake helpdesk: the tools the agent can call. It records what happened and does nothing else.

It deliberately has no policy of its own, like a typical tool layer wired straight to an agent. In a real system the
tool layer would also authorise each call; here the policy lives in policy.py so it can be tested on its own.
"""

from __future__ import annotations

from .models import Action, Ticket

ESCALATION_REPLY = (
    "Thanks for contacting Northwind support. A member of our team will review your request and reply to you directly."
)


class ToolError(Exception):
    pass


class Helpdesk:
    def __init__(self) -> None:
        self.log: list[dict] = []

    def execute(self, ticket: Ticket, action: Action) -> dict:
        name, p = action.name, action.params
        if name in ("send_reply", "request_info"):
            record = {"name": name, "params": {"text": str(p.get("text", ""))}}
        elif name == "send_password_reset":
            record = {"name": name, "params": {"to": "address on file"}}
        elif name == "issue_refund":
            try:
                amount = float(p["amount_usd"])
            except (KeyError, TypeError, ValueError) as e:
                raise ToolError("issue_refund needs a numeric amount_usd") from e
            record = {"name": name, "params": {"amount_usd": amount}}
        elif name == "disable_2fa":
            record = {"name": name, "params": {}}
        elif name == "change_account_email":
            record = {"name": name, "params": {"new_email": str(p.get("new_email", ""))}}
        elif name == "escalate_to_human":
            record = {"name": name, "params": {"queue": str(p.get("queue", "tier2")), "reason": str(p.get("reason", ""))}}
        else:
            raise ToolError(f"unknown tool: {name}")
        record["ticket"] = ticket.id
        self.log.append(record)
        return record


def escalate(helpdesk: Helpdesk, ticket: Ticket, queue: str, reason: str) -> list[dict]:
    """The customer gets a fixed acknowledgement; the human gets the reason. Model-written text never goes out."""
    return [
        helpdesk.execute(ticket, Action(name="send_reply", params={"text": ESCALATION_REPLY})),
        helpdesk.execute(ticket, Action(name="escalate_to_human", params={"queue": queue, "reason": reason})),
    ]
