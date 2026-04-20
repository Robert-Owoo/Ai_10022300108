"""
Student Name: Robert George Owoo
Index Number: 10022300108

Manual chunking implementation (word-based).

Design intent:
- Word-based chunking is simple, model-agnostic, and avoids tokenizer coupling.
- Overlap preserves context across boundaries (important for PDF narrative text).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cleaning import Document, normalize_text


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    parent_doc_id: str
    source: str
    title: str
    text: str
    start_word: int
    end_word: int


def chunk_document_words(
    doc: Document,
    *,
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    text = normalize_text(doc.text)
    words = text.split()

    if chunk_words <= 0:
        raise ValueError("chunk_words must be > 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be < chunk_words")

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_text = " ".join(words[start:end]).strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}__chunk_{chunk_idx}",
                    parent_doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    text=chunk_text,
                    start_word=start,
                    end_word=end,
                )
            )
            chunk_idx += 1
        if end >= len(words):
            break
        start = end - overlap_words

    return chunks


def chunk_corpus(
    docs: Iterable[Document],
    *,
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(
            chunk_document_words(d, chunk_words=chunk_words, overlap_words=overlap_words)
        )
    return out

