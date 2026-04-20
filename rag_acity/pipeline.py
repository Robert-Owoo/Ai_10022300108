"""
Student Name: Robert George Owoo
Index Number: 10022300108

Full RAG pipeline:
User Query → Retrieval → Context Selection → Prompt → LLM → Response

Includes logging at each stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bm25 import BM25Index
from .config import AppConfig
from .embedder import Embedder
from .feedback import FeedbackStore
from .llm import LLMClient, load_llm_config_from_env
from .logging_utils import StageLogger
from .prompting import build_prompt
from .retrieval import retrieve
from .vector_store import NumpyVectorStore


@dataclass
class LoadedIndex:
    vector_store: NumpyVectorStore
    bm25: BM25Index


def load_index(cfg: AppConfig) -> LoadedIndex:
    vs = NumpyVectorStore.load(emb_path=cfg.embeddings_path(), meta_path=cfg.metadata_path())
    bm25 = BM25Index.load(cfg.bm25_path())
    return LoadedIndex(vector_store=vs, bm25=bm25)


def run_rag(
    *,
    cfg: AppConfig,
    index: LoadedIndex,
    query: str,
    prompt_version: str,
    temperature: float,
    use_retrieval: bool,
    top_k: int | None = None,
    search_method: str = "hybrid",
    logger: StageLogger | None = None,
    run_path: Path | None = None,
) -> dict[str, Any]:
    """
    Returns a dict for UI display:
    - retrieved (list)
    - selected_context (list)
    - final_prompt (str)
    - answer (str)
    """
    tk = int(top_k) if top_k is not None else cfg.top_k
    embedder = Embedder(cfg.embedding_model_name)

    if logger:
        if run_path is None:
            run_path = logger.new_run_path()
        logger.log(
            run_path,
            "query",
            {
                "query": query,
                "use_retrieval": use_retrieval,
                "top_k": tk,
                "search_method": search_method,
            },
        )

    retrieved_for_prompt: list[dict] = []
    if use_retrieval:
        feedback_table = FeedbackStore(cfg.feedback_path()).load()
        qv = embedder.embed_query(query)
        hits = retrieve(
            mode=search_method,
            query=query,
            query_vec=qv,
            vector_store=index.vector_store,
            bm25=index.bm25,
            top_k=tk,
            alpha=cfg.hybrid_alpha,
            feedback_bonus=feedback_table,
            feedback_weight=cfg.feedback_weight,
        )
        retrieved_for_prompt = [
            {
                "idx": h.idx,
                "score": h.score,
                "score_vector": h.score_vector,
                "score_bm25": h.score_bm25,
                "chunk_id": index.vector_store.metadata[h.idx].get("chunk_id"),
                "parent_doc_id": index.vector_store.metadata[h.idx].get("parent_doc_id"),
                "source": index.vector_store.metadata[h.idx].get("source"),
                "title": index.vector_store.metadata[h.idx].get("title"),
                "text": index.vector_store.metadata[h.idx].get("text"),
            }
            for h in hits
        ]

        if logger and run_path:
            logger.log(
                run_path,
                "retrieval",
                {
                    "top_k": tk,
                    "search_method": search_method,
                    "hybrid_alpha": cfg.hybrid_alpha,
                    "feedback_weight": cfg.feedback_weight,
                    "hits": retrieved_for_prompt,
                },
            )

    prompt_res = build_prompt(
        user_query=query,
        retrieved_chunks=retrieved_for_prompt,
        max_context_words=cfg.max_context_words,
        prompt_version=prompt_version,
    )

    if logger and run_path:
        logger.log(
            run_path,
            "prompt",
            {
                "prompt_version": prompt_version,
                "selected_context": prompt_res.selected_context,
                "final_prompt": prompt_res.prompt,
            },
        )

    llm = LLMClient(load_llm_config_from_env())
    answer = llm.generate(prompt_res.prompt, temperature=temperature)

    if logger and run_path:
        logger.log(run_path, "generation", {"temperature": temperature, "answer": answer})

    return {
        "retrieved": retrieved_for_prompt,
        "selected_context": prompt_res.selected_context,
        "final_prompt": prompt_res.prompt,
        "answer": answer,
        "log_path": str(run_path) if run_path else None,
        "retrieval_settings": {
            "top_k": tk,
            "search_method": search_method,
            "use_retrieval": use_retrieval,
        },
    }

