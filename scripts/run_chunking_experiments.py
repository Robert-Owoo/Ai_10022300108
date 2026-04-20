"""
Student Name: Robert George Owoo
Index Number: 10022300108

Chunking impact experiment:
- Build two indexes with different (chunk_words, overlap)
- Run the same queries and measure a simple retrieval proxy:
  does top-1 hit come from the expected source? (budget_pdf vs election_csv)

This is not a perfect metric, but it produces evidence-based comparison
and is easy to reproduce.
"""

from __future__ import annotations

from dataclasses import replace

from rag_acity.bm25 import BM25Index
from rag_acity.chunking import chunk_corpus
from rag_acity.cleaning import budget_pdf_text_to_doc, load_election_csv_as_docs
from rag_acity.config import AppConfig
from rag_acity.data_sources import ensure_raw_datasets
from rag_acity.embedder import Embedder
from rag_acity.pdf_extract import extract_pdf_text
from rag_acity.retrieval import retrieve_hybrid
from rag_acity.vector_store import NumpyVectorStore


QUERIES = [
    ("Which party won in a specific constituency?", "election_csv"),
    ("What are key fiscal measures mentioned in the 2025 budget?", "budget_pdf"),
    ("Mention some revenue policy proposals in the 2025 budget statement.", "budget_pdf"),
    ("What does the election dataset say about a region/constituency result?", "election_csv"),
]


def build_in_memory_index(cfg: AppConfig) -> tuple[NumpyVectorStore, BM25Index]:
    paths = ensure_raw_datasets(cfg)
    election_docs = load_election_csv_as_docs(paths["election_csv"])
    budget_text = extract_pdf_text(paths["budget_pdf"])
    budget_doc = budget_pdf_text_to_doc(budget_text)

    chunks = chunk_corpus(
        [*election_docs, budget_doc],
        chunk_words=cfg.chunk_words,
        overlap_words=cfg.chunk_overlap_words,
    )
    embedder = Embedder(cfg.embedding_model_name)
    emb = embedder.embed_texts([c.text for c in chunks], batch_size=64)

    meta = [
        {
            "chunk_id": c.chunk_id,
            "parent_doc_id": c.parent_doc_id,
            "source": c.source,
            "title": c.title,
            "start_word": c.start_word,
            "end_word": c.end_word,
            "text": c.text,
        }
        for c in chunks
    ]
    vs = NumpyVectorStore(embeddings=emb, metadata=meta)
    bm25 = BM25Index.build([c.text for c in chunks])
    return vs, bm25


def eval_config(cfg: AppConfig) -> dict:
    vs, bm25 = build_in_memory_index(cfg)
    embedder = Embedder(cfg.embedding_model_name)

    correct = 0
    rows = []
    for q, expected_source in QUERIES:
        qv = embedder.embed_query(q)
        hits = retrieve_hybrid(
            query=q,
            query_vec=qv,
            vector_store=vs,
            bm25=bm25,
            top_k=3,
            alpha=cfg.hybrid_alpha,
        )
        top1_source = hits[0].metadata.get("source") if hits else None
        ok = top1_source == expected_source
        correct += int(ok)
        rows.append({"query": q, "expected": expected_source, "top1": top1_source, "ok": ok})

    return {
        "chunk_words": cfg.chunk_words,
        "overlap_words": cfg.chunk_overlap_words,
        "accuracy_top1_expected_source": correct / len(QUERIES),
        "details": rows,
    }


def main() -> None:
    base = AppConfig()
    cfg_a = replace(base, chunk_words=250, chunk_overlap_words=40)
    cfg_b = replace(base, chunk_words=600, chunk_overlap_words=120)

    res_a = eval_config(cfg_a)
    res_b = eval_config(cfg_b)

    print("=== Chunking Experiment Results (Top-1 expected source proxy) ===")
    for res in [res_a, res_b]:
        print()
        print(f"chunk_words={res['chunk_words']} overlap={res['overlap_words']}")
        print(f"accuracy={res['accuracy_top1_expected_source']:.2f}")
        for r in res["details"]:
            print(f"- ok={r['ok']} expected={r['expected']} top1={r['top1']} | {r['query']}")

    print()
    print("Interpretation guide:")
    print("- Smaller chunks may retrieve more precise local facts but can lose long-range context.")
    print("- Larger chunks preserve more narrative context but can dilute similarity and reduce precision.")


if __name__ == "__main__":
    main()

