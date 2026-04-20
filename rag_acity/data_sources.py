"""
Student Name: Robert George Owoo
Index Number: 10022300108

Download the provided datasets to local disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from .config import AppConfig


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download_file(url: str, dest_path: Path, *, timeout_s: int = 60) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    content = resp.content

    if dest_path.exists():
        existing = dest_path.read_bytes()
        if _sha256_bytes(existing) == _sha256_bytes(content):
            return dest_path

    dest_path.write_bytes(content)
    return dest_path


def ensure_raw_datasets(cfg: AppConfig) -> dict[str, Path]:
    """
    Returns paths to:
    - election CSV
    - budget PDF
    """
    cfg.data_raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cfg.data_raw_dir / "Ghana_Election_Result.csv"
    pdf_path = cfg.data_raw_dir / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"

    download_file(cfg.election_csv_url, csv_path)
    download_file(cfg.budget_pdf_url, pdf_path)

    return {"election_csv": csv_path, "budget_pdf": pdf_path}

