"""
Student Name: Robert George Owoo
Index Number: 10022300108

PDF extraction with a minimal dependency (pypdf).
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.replace("\x00", " ").strip()
        if text:
            parts.append(f"[PAGE {i+1}]\n{text}")
    return "\n\n".join(parts)

