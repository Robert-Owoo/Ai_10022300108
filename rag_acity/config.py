"""
Student Name: Robert George Owoo
Index Number: 10022300108
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    # Identity (required in every file by exam instructions)
    student_name: str = "Robert George Owoo"
    index_number: str = "10022300108"

    # Data sources (provided)
    election_csv_url: str = (
        "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/main/Ghana_Election_Result.csv"
    )
    # Note: MOFEP filename uses "and-Economic" (hyphen). The exam PDF URL without
    # that hyphen returns 404 as of 2026-04-16.
    budget_pdf_url: str = (
        "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    )

    # Local paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_raw_dir: Path = project_root / "data" / "raw"
    data_processed_dir: Path = project_root / "data" / "processed"
    indexes_dir: Path = project_root / "indexes"
    logs_dir: Path = project_root / "logs"

    # Default chunking strategy (can be overridden in scripts/UI)
    chunk_words: int = 350
    chunk_overlap_words: int = 60

    # Retrieval
    top_k: int = 6
    hybrid_alpha: float = 0.6  # 0..1 weight for vector; (1-alpha) for BM25
    feedback_weight: float = 0.15

    # Embeddings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Context window management for prompt (approx words budget)
    max_context_words: int = 1200

    # Index filenames (per chunking config)
    def index_prefix(self) -> str:
        return f"chunks_w{self.chunk_words}_o{self.chunk_overlap_words}"

    def embeddings_path(self) -> Path:
        return self.indexes_dir / f"{self.index_prefix()}_embeddings.npy"

    def metadata_path(self) -> Path:
        return self.indexes_dir / f"{self.index_prefix()}_metadata.jsonl"

    def bm25_path(self) -> Path:
        return self.indexes_dir / f"{self.index_prefix()}_bm25.json"

    def feedback_path(self) -> Path:
        return self.indexes_dir / "feedback.json"

