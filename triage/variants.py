"""The two one-call variants, built to be compared with the graph in graph.py.

- single:          one model call sees every runbook and its output is executed as given. This is the naive agent.
- single_guarded:  the same call, but the output goes through the same policy as the graph before anything runs.
                   No retrieval, no classify step, no second attempt.

Comparing the three separates what the policy buys from what the extra steps buy.
"""

from __future__ import annotations

from . import guard, policy, prompts
from .graph import TriageAgent
from .helpdesk import Helpdesk, ToolError, escalate
from .llm import LLM, LLMError, call_json
from .models import Result, SinglePlan, Ticket, derive_disposition
from .retrieval import Runbook

VARIANT_NAMES = ("single", "single_guarded", "multi")


class SingleAgent:
    def __init__(self, llm: LLM, runbooks: dict[str, Runbook], guarded: bool = False):
        self.llm = llm
        self.runbooks = runbooks
        self.guarded = guarded
        self.variant = "single_guarded" if guarded else "single"
        self._system = prompts.single_system(runbooks)
        self._texts = {rid: rb.text for rid, rb in runbooks.items()}

    def _failed(self, ticket: Ticket, hd: Helpdesk, reason: str, calls: int, tin=0, tout=0) -> Result:
        executed = escalate(hd, ticket, "tier2", reason) if self.guarded else []
        return Result(ticket.id, self.variant, derive_disposition(executed), executed=executed, reasons=[reason],
                      llm_calls=calls, input_tokens=tin, output_tokens=tout, error=reason)

    def run(self, ticket: Ticket) -> Result:
        hd = Helpdesk()
        try:
            call = call_json(self.llm, self._system, prompts.render_ticket(ticket), SinglePlan, max_tokens=900)
        except LLMError as e:
            return self._failed(ticket, hd, str(e), 1)
        plan = call.parsed
        if plan is None:
            return self._failed(ticket, hd, "invalid_output", call.calls, call.input_tokens, call.output_tokens)

        flags = guard.inspect(ticket.subject, ticket.body) if self.guarded else []
        reasons: list[str] = []
        if self.guarded:
            violations = policy.verify(ticket, plan, available=self._texts, flags=flags)
            early = policy.route(plan.category, "normal", 1.0)
            asked = next((a for a in plan.actions if a.name == "escalate_to_human"), None)
            if early or violations or asked:
                if early:
                    queue, why = early["queue"], early["reason"]
                elif violations:
                    queue, why = "tier2", "plan rejected: " + ", ".join(v.code for v in violations)
                else:
                    queue = str(asked.params.get("queue", "tier2"))
                    queue = queue if queue in policy.QUEUES else "tier2"
                    why = "the plan asked for a person"
                executed = escalate(hd, ticket, queue, why + (f" [guard flags: {','.join(flags)}]" if flags else ""))
                reasons = [why] + [v.code for v in violations]
                return self._result(ticket, plan, executed, reasons, flags, call)
        executed = []
        for action in plan.actions:
            try:
                executed.append(hd.execute(ticket, action))
            except ToolError as e:
                reasons.append(f"rejected by tool layer: {e}")
        return self._result(ticket, plan, executed, reasons, flags, call)

    def _result(self, ticket, plan, executed, reasons, flags, call) -> Result:
        return Result(
            ticket_id=ticket.id,
            variant=self.variant,
            disposition=derive_disposition(executed),
            category=plan.category,
            executed=executed,
            reasons=reasons,
            flags=flags,
            llm_calls=call.calls,
            input_tokens=call.input_tokens,
            output_tokens=call.output_tokens,
        )


def make_agent(name: str, llm: LLM, runbooks: dict[str, Runbook]):
    if name == "single":
        return SingleAgent(llm, runbooks, guarded=False)
    if name == "single_guarded":
        return SingleAgent(llm, runbooks, guarded=True)
    if name == "multi":
        return TriageAgent(llm, runbooks)
    raise ValueError(f"Unknown variant '{name}'. Use one of: {', '.join(VARIANT_NAMES)}")
