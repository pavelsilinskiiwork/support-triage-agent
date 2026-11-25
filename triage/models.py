"""Data shapes shared by every variant: the ticket, what the model proposes, and the outcome."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

CATEGORIES = ("billing", "account_access", "bug_report", "how_to", "outage", "security", "privacy", "other")
URGENCIES = ("low", "normal", "high", "critical")


class Customer(BaseModel):
    """Trusted metadata that comes from the helpdesk system, not from the customer's text."""

    customer_id: str = "C-0000"
    plan: str = "free"
    last_charge_days_ago: int | None = None
    last_charge_amount: float | None = None


class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    customer: Customer = Field(default_factory=Customer)


class Classification(BaseModel):
    category: str = "other"
    urgency: str = "normal"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        v = v.strip().lower()
        return v if v in CATEGORIES else "other"

    @field_validator("urgency")
    @classmethod
    def _known_urgency(cls, v: str) -> str:
        v = v.strip().lower()
        return v if v in URGENCIES else "normal"


class Action(BaseModel):
    """One tool call the model asks for. The name is a free string on purpose: unknown names must reach the policy."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class Plan(BaseModel):
    actions: list[Action] = Field(default_factory=list)
    cited_runbooks: list[str] = Field(default_factory=list)
    rationale: str = ""


class SinglePlan(Plan):
    """What the one-call variants ask the model for: a category, a disposition and the plan, all at once."""

    category: str = "other"
    disposition: str = "auto_resolve"

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        v = v.strip().lower()
        return v if v in CATEGORIES else "other"


@dataclass
class Result:
    ticket_id: str
    variant: str
    disposition: str  # auto_resolve | escalate | no_action
    category: str = "other"
    urgency: str = "normal"
    executed: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    trace: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def derive_disposition(executed: list[dict]) -> str:
    names = [e["name"] for e in executed]
    if "escalate_to_human" in names:
        return "escalate"
    return "auto_resolve" if names else "no_action"
