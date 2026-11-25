"""Runbook loading and a small BM25 index. No embeddings: with a few dozen runbooks, keyword ranking is enough
and it is deterministic, so retrieval can be tested without a model."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "runbooks"

STOP = frozenset(
    "a an and are as at be by can do for from has have how i if in is it its my of on or our so that the their "
    "this to was we what when with you your me us not no please hi hello thanks thank would could should".split()
)


@dataclass(frozen=True)
class Runbook:
    id: str
    title: str
    text: str


def load_runbooks(directory: Path | str = DEFAULT_DIR) -> dict[str, Runbook]:
    runbooks: dict[str, Runbook] = {}
    for path in sorted(Path(directory).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        match = re.match(r"---\n(.*?)\n---\n(.*)", raw, re.DOTALL)
        if not match:
            raise ValueError(f"{path.name}: missing front matter")
        meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
        rid = meta.get("id", "").strip()
        if not rid:
            raise ValueError(f"{path.name}: front matter has no id")
        runbooks[rid] = Runbook(rid, meta.get("title", rid).strip(), match.group(2).strip())
    return runbooks


def tokenize(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in STOP:
            continue
        if len(tok) > 3 and tok.endswith("s"):
            tok = tok[:-1]
        out.append(tok)
    return out


class Index:
    def __init__(self, runbooks: dict[str, Runbook], k1: float = 1.5, b: float = 0.75):
        self.runbooks = runbooks
        self.k1, self.b = k1, b
        self._docs = {rid: Counter(tokenize(f"{rb.title} {rb.title} {rb.text}")) for rid, rb in runbooks.items()}
        self._len = {rid: sum(c.values()) for rid, c in self._docs.items()}
        self._avg = sum(self._len.values()) / max(len(self._len), 1)
        df: Counter = Counter()
        for counts in self._docs.values():
            df.update(counts.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 3) -> list[tuple[Runbook, float]]:
        terms = tokenize(query)
        scored = []
        for rid, counts in self._docs.items():
            score = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self._len[rid] / self._avg)
                score += self._idf[t] * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((self.runbooks[rid], score))
        scored.sort(key=lambda x: (-x[1], x[0].id))
        return scored[:k]
