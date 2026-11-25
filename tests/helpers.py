import json
from collections.abc import Callable

from triage.llm import Reply


class ScriptedLLM:
    """Returns prepared answers in order and records every call. A string is a reply, an Exception is raised."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, *answers: str | Exception | Callable[[str, str], str]):
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, max_tokens: int = 800) -> Reply:
        self.calls.append((system, user))
        if not self.answers:
            raise AssertionError("ScriptedLLM ran out of answers")
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        if callable(a):
            a = a(system, user)
        return Reply(a, input_tokens=100, output_tokens=20)


def classify(category="billing", urgency="normal", confidence=0.9) -> str:
    return json.dumps({"category": category, "urgency": urgency, "confidence": confidence, "summary": "x"})


def plan(*actions, cited=("billing-refund-policy",), rationale="r") -> str:
    return json.dumps({"actions": [{"name": n, "params": p} for n, p in actions], "cited_runbooks": list(cited),
                       "rationale": rationale})


def single(category, disposition, *actions, cited=("billing-refund-policy",)) -> str:
    d = json.loads(plan(*actions, cited=cited))
    return json.dumps({"category": category, "disposition": disposition, **d})


def reply(text="Thanks. See https://northwind.example/legal/refunds"):
    return ("send_reply", {"text": text})


def refund(amount):
    return ("issue_refund", {"amount_usd": amount})
