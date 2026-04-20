"""
Student Name: Robert George Owoo
Index Number: 10022300108

Innovation component (Part G): retrieval feedback loop.

Users can upvote/downvote specific retrieved chunks. We persist a small score table
and apply it as a bonus/penalty during retrieval fusion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeedbackStore:
    path: Path

    def load(self) -> dict[str, float]:
        if not self.path.exists():
            return {}
        try:
            return {k: float(v) for k, v in json.loads(self.path.read_text(encoding="utf-8")).items()}
        except Exception:
            return {}

    def save(self, table: dict[str, float]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    def update(self, chunk_id: str, delta: float) -> dict[str, float]:
        table = self.load()
        table[chunk_id] = float(table.get(chunk_id, 0.0) + delta)
        # keep values in a reasonable range
        table[chunk_id] = max(-2.0, min(2.0, table[chunk_id]))
        self.save(table)
        return table

    def bonus(self, chunk_id: str) -> float:
        return float(self.load().get(chunk_id, 0.0))

