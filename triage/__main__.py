"""Command line: run one ticket through a variant, or print the graph.

    python -m triage run examples/refund_ok.json --provider standin
    python -m triage run examples/refund_ok.json --provider anthropic --variant multi
    python -m triage graph
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import get_llm
from .models import Ticket
from .retrieval import load_runbooks
from .variants import VARIANT_NAMES, make_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m triage", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="run one ticket (a JSON file) through a variant")
    run.add_argument("ticket", type=Path)
    run.add_argument("--provider", default=None, help="anthropic | bedrock | groq | standin (default: $LLM_PROVIDER)")
    run.add_argument("--model", default=None)
    run.add_argument("--variant", default="multi", choices=VARIANT_NAMES)
    sub.add_parser("graph", help="print the multi-step graph as a Mermaid diagram")
    args = parser.parse_args(argv)

    runbooks = load_runbooks()
    if args.cmd == "graph":
        from .graph import TriageAgent

        print(TriageAgent(get_llm("standin"), runbooks).mermaid())
        return 0

    ticket = Ticket.model_validate(json.loads(args.ticket.read_text(encoding="utf-8")))
    agent = make_agent(args.variant, get_llm(args.provider, args.model), runbooks)
    result = agent.run(ticket)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
