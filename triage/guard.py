"""Heuristic detection of instruction-like text in a ticket.

This is a tripwire, not the defence. It only runs on English, it is easy to evade, and a clean result proves nothing.
What it buys: when it does fire, the policy refuses privileged actions for that ticket (see policy.py), and a human
sees the flags. The real limits on what the agent can do come from the policy and the trusted metadata.
"""

from __future__ import annotations

import re
import unicodedata

PATTERNS: dict[str, re.Pattern] = {
    "ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|your)\b"
        r".{0,30}\b(instructions?|rules?|prompts?|guidelines?)\b"
    ),
    "role_override": re.compile(
        r"\byou are (now|no longer)\b|\b(developer|admin|debug|god|refund) mode\b|\bact as\b.{0,20}\b(admin|administrator|root)\b"
    ),
    "fake_markup": re.compile(
        r"</?\s*(system|assistant|ticket|runbook|runbooks|metadata|instructions?)\s*>|\[/?(inst|system)\]|<\|im_(start|end)\|>"
    ),
    "prompt_probe": re.compile(
        r"\b(reveal|print|show|repeat|output|leak|display)\b.{0,40}"
        r"\b(system prompt|hidden (rules|instructions)|internal (rules|instructions|reference)|your (instructions|prompt))\b"
    ),
    "tool_names": re.compile(
        r"\b(issue_refund|disable_2fa|change_account_email|send_password_reset|escalate_to_human|send_reply|request_info)\b"
    ),
    "json_actions": re.compile(r"\"(actions|amount_usd|cited_runbooks)\"\s*:"),
    "link_request": re.compile(r"\b(include|add|insert|put|paste|tell the customer)\b.{0,60}\b(link|url|https?://)"),
    "encoded_blob": re.compile(r"[a-z0-9+/]{60,}={0,2}", re.IGNORECASE),
}


def normalise(text: str) -> str:
    """NFKC, drop zero-width and other format characters, collapse whitespace, lower-case."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", text).lower()


def inspect(subject: str, body: str) -> list[str]:
    raw = f"{subject}\n{body}"
    flags = {name for name, pat in PATTERNS.items() if pat.search(normalise(raw))}
    if PATTERNS["encoded_blob"].search(raw):  # case matters for blobs; run on the raw text too
        flags.add("encoded_blob")
    return sorted(flags)
