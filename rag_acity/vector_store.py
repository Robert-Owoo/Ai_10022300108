"""
Student Name: Robert George Owoo
Index Number: 10022300108

Minimal vector store:
- Stores normalized embeddings (float32) in a .npy file
- Stores per-chunk metadata in JSONL
- Retrieves top-k by cosine similarity (dot product because vectors are normalized)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VectorHit:
    idx: int
    score: float
    metadata: dict[str, Any]


class NumpyVectorStore:
    def __init__(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]):
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("metadata length must match embeddings rows")
        self.embeddings = embeddings.astype(np.float32, copy=False)
        self.metadata = metadata

    @staticmethod
    def save(embeddings: np.ndarray, metadata: list[dict[str, Any]], *, emb_path: Path, meta_path: Path) -> None:
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, embeddings.astype(np.float32, copy=False))
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            for m in metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    @staticmethod
    def load(*, emb_path: Path, meta_path: Path) -> "NumpyVectorStore":
        embeddings = np.load(emb_path)
        metadata: list[dict[str, Any]] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                metadata.append(json.loads(line))
        return NumpyVectorStore(embeddings=embeddings, metadata=metadata)

    def search(self, query_vec: np.ndarray, *, top_k: int) -> list[VectorHit]:
        q = query_vec.astype(np.float32, copy=False)
        if q.ndim != 1:
            raise ValueError("query_vec must be 1D")

        scores = self.embeddings @ q  # cosine because normalized
        k = min(top_k, scores.shape[0])
        if k <= 0:
            return []

        # Partial top-k for speed
        idxs = np.argpartition(-scores, kth=k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]

        hits: list[VectorHit] = []
        for i in idxs.tolist():
            hits.append(VectorHit(idx=i, score=float(scores[i]), metadata=self.metadata[i]))
        return hits

