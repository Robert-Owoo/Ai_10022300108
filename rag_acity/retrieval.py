"""
Student Name: Robert George Owoo
Index Number: 10022300108

Retrieval:
- Vector similarity (cosine, dense embeddings)
- BM25 keyword retrieval
- Hybrid fusion (fixes common failure cases)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .vector_store import NumpyVectorStore, VectorHit


@dataclass(frozen=True)
class RetrievalHit:
    idx: int
    score: float
    score_vector: float
    score_bm25: float
    metadata: dict[str, Any]


def _minmax_norm(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores.keys()}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def retrieve_hybrid(
    *,
    query: str,
    query_vec: np.ndarray,
    vector_store: NumpyVectorStore,
    bm25: BM25Index,
    top_k: int,
    alpha: float,
    candidate_k: int | None = None,
    feedback_bonus: dict[str, float] | None = None,
    feedback_weight: float = 0.15,
) -> list[RetrievalHit]:
    """
    Hybrid retrieval via score fusion:
    base_score = alpha * norm(vector_score) + (1-alpha) * norm(bm25_score)

    Innovation (Part G): optional feedback bonus:
    score = base_score + feedback_weight * bonus(chunk_id)
    """
    alpha = float(alpha)
    alpha = max(0.0, min(1.0, alpha))
    cand_k = candidate_k or max(top_k * 5, 30)
    feedback_weight = float(feedback_weight)
    if feedback_bonus is None:
        feedback_bonus = {}

    vec_hits: list[VectorHit] = vector_store.search(query_vec, top_k=cand_k)
    bm25_hits = bm25.score(query, top_k=cand_k)

    vec_scores = {h.idx: float(h.score) for h in vec_hits}
    bm_scores = {i: float(s) for i, s in bm25_hits}

    vec_n = _minmax_norm(vec_scores)
    bm_n = _minmax_norm(bm_scores)

    all_ids = set(vec_scores.keys()) | set(bm_scores.keys())
    fused: list[RetrievalHit] = []
    for i in all_ids:
        sv = vec_n.get(i, 0.0)
        sb = bm_n.get(i, 0.0)
        base_score = alpha * sv + (1.0 - alpha) * sb
        chunk_id = str(vector_store.metadata[i].get("chunk_id", ""))
        bonus = float(feedback_bonus.get(chunk_id, 0.0))
        score = base_score + feedback_weight * bonus
        fused.append(
            RetrievalHit(
                idx=i,
                score=float(score),
                score_vector=float(vec_scores.get(i, 0.0)),
                score_bm25=float(bm_scores.get(i, 0.0)),
                metadata=vector_store.metadata[i],
            )
        )

    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]


def _apply_feedback(
    *,
    vector_store: NumpyVectorStore,
    idx: int,
    base_score: float,
    vec_score: float,
    bm25_score: float,
    feedback_bonus: dict[str, float],
    feedback_weight: float,
) -> RetrievalHit:
    chunk_id = str(vector_store.metadata[idx].get("chunk_id", ""))
    bonus = float(feedback_bonus.get(chunk_id, 0.0))
    score = float(base_score) + float(feedback_weight) * bonus
    return RetrievalHit(
        idx=idx,
        score=score,
        score_vector=float(vec_score),
        score_bm25=float(bm25_score),
        metadata=vector_store.metadata[idx],
    )


def retrieve_vector_only(
    *,
    query_vec: np.ndarray,
    vector_store: NumpyVectorStore,
    bm25: BM25Index,
    top_k: int,
    candidate_k: int | None = None,
    feedback_bonus: dict[str, float] | None = None,
    feedback_weight: float = 0.15,
) -> list[RetrievalHit]:
    """Dense retrieval only (BM25 index unused except for shape compatibility)."""
    _ = bm25
    cand_k = candidate_k or max(top_k * 5, 30)
    if feedback_bonus is None:
        feedback_bonus = {}
    vec_hits = vector_store.search(query_vec, top_k=cand_k)
    vec_scores = {h.idx: float(h.score) for h in vec_hits}
    vec_n = _minmax_norm(vec_scores)
    fused: list[RetrievalHit] = []
    for i in vec_scores:
        fused.append(
            _apply_feedback(
                vector_store=vector_store,
                idx=i,
                base_score=vec_n.get(i, 0.0),
                vec_score=vec_scores[i],
                bm25_score=0.0,
                feedback_bonus=feedback_bonus,
                feedback_weight=feedback_weight,
            )
        )
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]


def retrieve_bm25_only(
    *,
    query: str,
    vector_store: NumpyVectorStore,
    bm25: BM25Index,
    top_k: int,
    candidate_k: int | None = None,
    feedback_bonus: dict[str, float] | None = None,
    feedback_weight: float = 0.15,
) -> list[RetrievalHit]:
    """Keyword BM25 only (embeddings unused)."""
    cand_k = candidate_k or max(top_k * 5, 30)
    if feedback_bonus is None:
        feedback_bonus = {}
    bm25_hits = bm25.score(query, top_k=cand_k)
    bm_scores = {i: float(s) for i, s in bm25_hits}
    bm_n = _minmax_norm(bm_scores)
    fused: list[RetrievalHit] = []
    for i in bm_scores:
        fused.append(
            _apply_feedback(
                vector_store=vector_store,
                idx=i,
                base_score=bm_n.get(i, 0.0),
                vec_score=0.0,
                bm25_score=bm_scores[i],
                feedback_bonus=feedback_bonus,
                feedback_weight=feedback_weight,
            )
        )
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused[:top_k]


def retrieve(
    *,
    mode: str,
    query: str,
    query_vec: np.ndarray,
    vector_store: NumpyVectorStore,
    bm25: BM25Index,
    top_k: int,
    alpha: float,
    feedback_bonus: dict[str, float] | None = None,
    feedback_weight: float = 0.15,
) -> list[RetrievalHit]:
    m = (mode or "hybrid").strip().lower()
    if m == "vector":
        return retrieve_vector_only(
            query_vec=query_vec,
            vector_store=vector_store,
            bm25=bm25,
            top_k=top_k,
            feedback_bonus=feedback_bonus,
            feedback_weight=feedback_weight,
        )
    if m == "bm25":
        return retrieve_bm25_only(
            query=query,
            vector_store=vector_store,
            bm25=bm25,
            top_k=top_k,
            feedback_bonus=feedback_bonus,
            feedback_weight=feedback_weight,
        )
    return retrieve_hybrid(
        query=query,
        query_vec=query_vec,
        vector_store=vector_store,
        bm25=bm25,
        top_k=top_k,
        alpha=alpha,
        feedback_bonus=feedback_bonus,
        feedback_weight=feedback_weight,
    )

