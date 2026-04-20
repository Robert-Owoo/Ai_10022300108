"""
Student Name: Robert George Owoo
Index Number: 10022300108

Builds the RAG index (manual pipeline):
download → extract/clean → chunk → embed → save vector store + BM25 index
"""

from __future__ import annotations

from rag_acity.bm25 import BM25Index
from rag_acity.chunking import chunk_corpus
from rag_acity.cleaning import (
    budget_pdf_text_to_doc,
    load_election_csv_as_docs,
    save_processed_docs,
)
from rag_acity.config import AppConfig
from rag_acity.data_sources import ensure_raw_datasets
from rag_acity.embedder import Embedder
from rag_acity.pdf_extract import extract_pdf_text
from rag_acity.vector_store import NumpyVectorStore


def main() -> None:
    cfg = AppConfig()

    paths = ensure_raw_datasets(cfg)
    election_docs = load_election_csv_as_docs(paths["election_csv"])
    budget_text = extract_pdf_text(paths["budget_pdf"])
    budget_doc = budget_pdf_text_to_doc(budget_text)

    processed_path = cfg.data_processed_dir / "processed_docs.jsonl"
    save_processed_docs([*election_docs, budget_doc], processed_path)

    chunks = chunk_corpus(
        [*election_docs, budget_doc],
        chunk_words=cfg.chunk_words,
        overlap_words=cfg.chunk_overlap_words,
    )

    embedder = Embedder(cfg.embedding_model_name)
    embeddings = embedder.embed_texts([c.text for c in chunks], batch_size=64)

    metadata = [
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

    NumpyVectorStore.save(
        embeddings,
        metadata,
        emb_path=cfg.embeddings_path(),
        meta_path=cfg.metadata_path(),
    )

    bm25 = BM25Index.build([c.text for c in chunks])
    bm25.save(cfg.bm25_path())

    print("Index build complete.")
    print(f"Chunks: {len(chunks)}")
    print(f"Embeddings: {cfg.embeddings_path()}")
    print(f"Metadata: {cfg.metadata_path()}")
    print(f"BM25: {cfg.bm25_path()}")


if __name__ == "__main__":
    main()

