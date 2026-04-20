"""
Student Name: Robert George Owoo
Index Number: 10022300108

Embedding pipeline (manual):
- Loads a SentenceTransformer model
- Encodes text -> dense vectors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Embedder:
    model_name: str
    _model: SentenceTransformer | None = None

    def load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        model = self.load()
        vecs = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        model = self.load()
        vec = model.encode([query], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32)

