"""
Student Name: Robert George Owoo
Index Number: 10022300108

Keyword retrieval (BM25) implemented manually.
This enables hybrid retrieval (keyword + vector) and fixes some failure cases
where dense embeddings underperform on exact-number or keyword queries.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text)]


@dataclass(frozen=True)
class BM25Index:
    # Stored as compact dicts for JSON persistence
    doc_len: list[int]
    avgdl: float
    df: dict[str, int]
    tf: list[dict[str, int]]  # per doc token counts
    k1: float = 1.5
    b: float = 0.75

    @staticmethod
    def build(texts: Iterable[str], *, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        doc_len: list[int] = []

        for text in texts:
            tokens = tokenize(text)
            doc_len.append(len(tokens))
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            tf.append(counts)
            for t in counts.keys():
                df[t] = df.get(t, 0) + 1

        avgdl = (sum(doc_len) / len(doc_len)) if doc_len else 0.0
        return BM25Index(doc_len=doc_len, avgdl=avgdl, df=df, tf=tf, k1=k1, b=b)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        obj: dict[str, Any] = {
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "df": self.df,
            "tf": self.tf,
            "k1": self.k1,
            "b": self.b,
        }
        path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "BM25Index":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return BM25Index(
            doc_len=list(obj["doc_len"]),
            avgdl=float(obj["avgdl"]),
            df=dict(obj["df"]),
            tf=list(obj["tf"]),
            k1=float(obj.get("k1", 1.5)),
            b=float(obj.get("b", 0.75)),
        )

    def score(self, query: str, *, top_k: int) -> list[tuple[int, float]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        N = len(self.tf)
        scores = [0.0] * N
        for t in tokens:
            n_qi = self.df.get(t, 0)
            if n_qi == 0:
                continue
            # Standard BM25 idf
            idf = math.log(1.0 + (N - n_qi + 0.5) / (n_qi + 0.5))
            for i in range(N):
                f = self.tf[i].get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / (self.avgdl + 1e-9)))
                scores[i] += idf * ((f * (self.k1 + 1)) / (denom + 1e-9))

        k = min(top_k, N)
        if k <= 0:
            return []
        idxs = sorted(range(N), key=lambda i: scores[i], reverse=True)[:k]
        return [(i, float(scores[i])) for i in idxs if scores[i] > 0]

