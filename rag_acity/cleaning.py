"""
Student Name: Robert George Owoo
Index Number: 10022300108

Data cleaning for:
- Election CSV (tabular -> canonical text rows)
- Budget PDF text (whitespace cleanup)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str  # "election_csv" | "budget_pdf"
    title: str
    text: str


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RE.sub(" ", s)
    s = _NL_RE.sub("\n\n", s)
    return s.strip()


def load_election_csv_as_docs(csv_path: Path) -> list[Document]:
    df = pd.read_csv(csv_path)
    # Basic cleaning: drop fully empty rows/cols, normalize column names
    df = df.dropna(how="all")
    df.columns = [str(c).strip() for c in df.columns]

    # Convert each row into a text “fact record”
    docs: list[Document] = []
    for idx, row in df.iterrows():
        fields = []
        for col in df.columns:
            val = row.get(col)
            if pd.isna(val):
                continue
            sval = str(val).strip()
            if sval == "":
                continue
            fields.append(f"{col}: {sval}")

        if not fields:
            continue

        text = normalize_text(" | ".join(fields))
        docs.append(
            Document(
                doc_id=f"election_row_{idx}",
                source="election_csv",
                title="Ghana Election Result (row)",
                text=text,
            )
        )

    return docs


def budget_pdf_text_to_doc(text: str) -> Document:
    cleaned = normalize_text(text)
    return Document(
        doc_id="budget_2025_pdf",
        source="budget_pdf",
        title="2025 Budget Statement and Economic Policy (MOFEP)",
        text=cleaned,
    )


def save_processed_docs(docs: Iterable[Document], out_path: Path) -> None:
    """
    Saves as JSONL with fields: doc_id, source, title, text.
    """
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {"doc_id": d.doc_id, "source": d.source, "title": d.title, "text": d.text},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_processed_docs(processed_jsonl: Path) -> list[Document]:
    import json

    docs: list[Document] = []
    with processed_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            docs.append(Document(**obj))
    return docs

