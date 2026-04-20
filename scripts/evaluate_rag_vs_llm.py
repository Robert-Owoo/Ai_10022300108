"""
Student Name: Robert George Owoo
Index Number: 10022300108

Evidence-based comparison:
RAG (with retrieval) vs Pure LLM (no retrieval).

This script:
- Loads the built index
- Runs the same queries twice (to assess consistency)
- Writes a JSONL report with prompts, retrieved hits, and outputs

Note: Requires OPENAI_API_KEY for generation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from rag_acity.config import AppConfig
from rag_acity.logging_utils import StageLogger
from rag_acity.pipeline import load_index, run_rag


ADVERSARIAL_QUERIES = [
    # Ambiguous
    "Who won the election and why?",
    # Misleading/incomplete (likely not explicitly answerable from provided docs)
    "What was Ghana's GDP growth rate in 2010 according to the budget statement?",
]


NORMAL_QUERIES = [
    "Summarize two fiscal policy measures mentioned in the 2025 budget statement.",
    "From the election CSV, provide an example of a record format and explain what fields it contains.",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_suite(cfg: AppConfig) -> None:
    index = load_index(cfg)
    logger = StageLogger(cfg.logs_dir)
    out_rows: list[dict] = []

    all_queries = NORMAL_QUERIES + ADVERSARIAL_QUERIES

    for q in all_queries:
        for trial in [1, 2]:
            for mode in ["rag", "pure_llm"]:
                use_retrieval = mode == "rag"
                run_path = logger.new_run_path()
                t0 = time.time()
                res = run_rag(
                    cfg=cfg,
                    index=index,
                    query=q,
                    prompt_version="v2",
                    temperature=0.2,
                    use_retrieval=use_retrieval,
                    logger=logger,
                    run_path=run_path,
                )
                dt = time.time() - t0
                out_rows.append(
                    {
                        "query": q,
                        "trial": trial,
                        "mode": mode,
                        "latency_s": dt,
                        "retrieved_count": len(res["retrieved"]),
                        "answer": res["answer"],
                        "final_prompt": res["final_prompt"],
                        "retrieved": res["retrieved"],
                        "log_path": res["log_path"],
                    }
                )
                print(f"[{mode}] trial={trial} done: {q[:60]}...")

    out_path = cfg.project_root / "experiments" / "eval_results.jsonl"
    write_jsonl(out_path, out_rows)
    print(f"Saved: {out_path}")


def main() -> None:
    cfg = AppConfig()
    run_suite(cfg)


if __name__ == "__main__":
    main()

