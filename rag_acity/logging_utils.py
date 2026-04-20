"""
Student Name: Robert George Owoo
Index Number: 10022300108

Structured logging for exam requirements:
- log at each stage of RAG pipeline
- store retrieved docs, scores, and final prompt
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageLogger:
    logs_dir: Path

    def new_run_path(self) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        return self.logs_dir / f"run_{ts}.jsonl"

    def log(self, run_path: Path, stage: str, payload: dict[str, Any]) -> None:
        rec = {"ts": time.time(), "stage": stage, "payload": payload}
        with run_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

