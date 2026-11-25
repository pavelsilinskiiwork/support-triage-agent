"""The multi-step agent as a LangGraph state graph.

    guard -> classify -> [escalate | retrieve -> plan -> [escalate | verify -> [execute | plan (once) | escalate]]]

Only classify and plan call the model. Guard, retrieve, verify, execute and escalate are plain code. Whatever the
model says, a ticket only gets automatic actions after verify() accepts the plan.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from . import guard, policy, prompts
from .helpdesk import Helpdesk, escalate
from .llm import LLM, LLMError, call_json
from .models import Classification, Plan, Result, Ticket, derive_disposition
from .retrieval import Index, Runbook

MAX_REPLANS = 1
TOP_K = 3


class State(TypedDict, total=False):
    ticket: Ticket
    helpdesk: Helpdesk
    flags: list[str]
    category: str
    urgency: str
    confidence: float
    summary: str
    retrieved: list[str]
    plan: Plan | None
    violations: list[policy.Violation]
    replans: int
    verdict: str
    escalation: dict | None
    executed: list[dict]
    reasons: list[str]
    error: str | None
    llm_calls: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]
    trace: Annotated[list[str], operator.add]


def build_graph(llm: LLM, runbooks: dict[str, Runbook]):
    index = Index(runbooks)

    def guard_node(state: State) -> dict:
        t = state["ticket"]
        flags = guard.inspect(t.subject, t.body)
        return {"flags": flags, "trace": [f"guard: {','.join(flags) or 'clean'}"]}

    def classify_node(state: State) -> dict:
        t = state["ticket"]
        try:
            call = call_json(llm, prompts.classify_system(), prompts.render_ticket(t), Classification, max_tokens=300)
        except LLMError as e:
            esc = policy.route("other", "normal", 0.0, "llm_error")
            return {"category": "other", "urgency": "normal", "confidence": 0.0, "escalation": esc, "error": str(e),
                    "llm_calls": 1, "trace": ["classify: llm error"]}
        usage = {"llm_calls": call.calls, "input_tokens": call.input_tokens, "output_tokens": call.output_tokens}
        c = call.parsed
        if c is None:
            esc = policy.route("other", "normal", 0.0, "invalid_json")
            return {"category": "other", "urgency": "normal", "confidence": 0.0, "escalation": esc, **usage,
                    "trace": ["classify: invalid output"]}
        esc = policy.route(c.category, c.urgency, c.confidence)
        return {"category": c.category, "urgency": c.urgency, "confidence": c.confidence, "summary": c.summary,
                "escalation": esc, **usage,
                "trace": [f"classify: {c.category}/{c.urgency} conf={c.confidence:.2f}"]}

    def after_classify(state: State) -> str:
        return "escalate" if state.get("escalation") else "retrieve"

    def retrieve_node(state: State) -> dict:
        t = state["ticket"]
        hits = index.search(f"{state['category']} {t.subject} {t.body[:1500]}", k=TOP_K)
        ids = [rb.id for rb, _ in hits]
        return {"retrieved": ids, "trace": [f"retrieve: {','.join(ids) or 'nothing'}"]}

    def plan_node(state: State) -> dict:
        t = state["ticket"]
        shown = [runbooks[i] for i in state.get("retrieved", [])]
        user = (
            f"{prompts.render_ticket(t)}\n\nClassification: category={state['category']} urgency={state['urgency']}\n\n"
            f"Runbooks:\n{prompts.render_runbooks(shown)}"
        )
        feedback = state.get("violations") or []
        if feedback and state.get("replans", 0) > 0:
            user += "\n\nYour previous plan was rejected because:\n" + "\n".join(f"- {v.description}" for v in feedback)
            user += "\nProduce a corrected plan. Change only what is needed to fix these problems."
        try:
            call = call_json(llm, prompts.plan_system(), user, Plan, max_tokens=900)
        except LLMError as e:
            return {"plan": None, "escalation": {"queue": "tier2", "reason": "model call failed while planning"},
                    "error": str(e), "llm_calls": 1, "trace": ["plan: llm error"]}
        names = [a.name for a in call.parsed.actions] if call.parsed else []
        return {"plan": call.parsed, "llm_calls": call.calls, "input_tokens": call.input_tokens,
                "output_tokens": call.output_tokens, "trace": [f"plan: {','.join(names) or 'invalid'}"]}

    def after_plan(state: State) -> str:
        return "escalate" if state.get("escalation") else "verify"

    def verify_node(state: State) -> dict:
        plan = state.get("plan")
        shown = {i: runbooks[i].text for i in state.get("retrieved", [])}
        violations = policy.verify(state["ticket"], plan, available=shown, flags=state.get("flags", []))
        codes = [v.code for v in violations]
        wants_human = plan is not None and any(a.name == "escalate_to_human" for a in plan.actions)
        if wants_human:
            asked = next(a for a in plan.actions if a.name == "escalate_to_human")
            queue = str(asked.params.get("queue", "tier2"))
            queue = queue if queue in policy.QUEUES else "tier2"
            return {"verdict": "escalate", "violations": violations,
                    "escalation": {"queue": queue, "reason": "the plan asked for a person"},
                    "trace": [f"verify: plan asked for escalation to {queue}"]}
        if not violations:
            return {"verdict": "ok", "violations": [], "trace": ["verify: ok"]}
        if not policy.is_terminal(violations) and state.get("replans", 0) < MAX_REPLANS:
            return {"verdict": "replan", "violations": violations, "replans": state.get("replans", 0) + 1,
                    "trace": [f"verify: rejected ({','.join(codes)}), replanning"]}
        reason = "plan rejected: " + ", ".join(codes)
        return {"verdict": "escalate", "violations": violations, "escalation": {"queue": "tier2", "reason": reason},
                "trace": [f"verify: rejected ({','.join(codes)}), escalating"]}

    def after_verify(state: State) -> str:
        return {"ok": "execute", "replan": "plan"}.get(state["verdict"], "escalate")

    def execute_node(state: State) -> dict:
        t, hd = state["ticket"], state["helpdesk"]
        executed = [hd.execute(t, a) for a in state["plan"].actions]
        return {"executed": executed, "reasons": [], "trace": [f"execute: {','.join(e['name'] for e in executed)}"]}

    def escalate_node(state: State) -> dict:
        t, hd = state["ticket"], state["helpdesk"]
        esc = state.get("escalation") or {"queue": "tier2", "reason": "unspecified"}
        flags = state.get("flags") or []
        reason = esc["reason"] + (f" [guard flags: {','.join(flags)}]" if flags else "")
        executed = escalate(hd, t, esc["queue"], reason)
        reasons = [esc["reason"]] + [v.code for v in state.get("violations") or []]
        return {"executed": executed, "reasons": reasons, "trace": [f"escalate: {esc['queue']}"]}

    g = StateGraph(State)
    for name, fn in [("guard", guard_node), ("classify", classify_node), ("retrieve", retrieve_node),
                     ("plan", plan_node), ("verify", verify_node), ("execute", execute_node),
                     ("escalate", escalate_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard")
    g.add_edge("guard", "classify")
    g.add_conditional_edges("classify", after_classify, {"escalate": "escalate", "retrieve": "retrieve"})
    g.add_edge("retrieve", "plan")
    g.add_conditional_edges("plan", after_plan, {"escalate": "escalate", "verify": "verify"})
    g.add_conditional_edges("verify", after_verify, {"execute": "execute", "plan": "plan", "escalate": "escalate"})
    g.add_edge("execute", END)
    g.add_edge("escalate", END)
    return g.compile()


class TriageAgent:
    """The multi-step variant. One compiled graph, safe to reuse across tickets and threads."""

    variant = "multi"

    def __init__(self, llm: LLM, runbooks: dict[str, Runbook]):
        self.graph = build_graph(llm, runbooks)

    def run(self, ticket: Ticket) -> Result:
        final: dict[str, Any] = self.graph.invoke({"ticket": ticket, "helpdesk": Helpdesk(), "replans": 0})
        executed = final.get("executed", [])
        return Result(
            ticket_id=ticket.id,
            variant=self.variant,
            disposition=derive_disposition(executed),
            category=final.get("category", "other"),
            urgency=final.get("urgency", "normal"),
            executed=executed,
            reasons=final.get("reasons", []),
            flags=final.get("flags", []),
            llm_calls=final.get("llm_calls", 0),
            input_tokens=final.get("input_tokens", 0),
            output_tokens=final.get("output_tokens", 0),
            trace=final.get("trace", []),
            error=final.get("error"),
        )

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()
