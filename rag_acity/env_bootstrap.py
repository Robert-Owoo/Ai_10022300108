"""
Student Name: Robert George Owoo
Index Number: 10022300108

Load `.env` from likely project locations so OPENAI_API_KEY is found when Streamlit
or scripts run from different working directories.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_dotenv_files() -> list[Path]:
    """
    Load `.env` from:
    1) Repo root (folder containing `rag_acity/`) — **override=True** so values here
       win over stale `OPENAI_*` variables left in the shell from an old session.
    2) Current working directory — **override=False** (fill only missing keys).
    """
    candidates = [
        Path(__file__).resolve().parents[1] / ".env",
        Path.cwd() / ".env",
    ]
    loaded: list[Path] = []
    seen: set[str] = set()
    for i, p in enumerate(candidates):
        try:
            rp = str(p.resolve())
        except OSError:
            continue
        if rp in seen or not p.is_file():
            continue
        # First file (project root) should override process env — common student issue:
        # PowerShell still has an old invalid OPENAI_API_KEY while .env has the new key.
        load_dotenv(p, override=(i == 0))
        loaded.append(p)
        seen.add(rp)
    return loaded


def normalize_api_key(raw: str) -> str:
    """Strip whitespace, UTF-8 BOM, and a single pair of surrounding quotes from .env mistakes."""
    s = raw.strip()
    if s.startswith("\ufeff"):
        s = s.lstrip("\ufeff").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def sync_normalized_api_key_to_environ() -> None:
    """Rewrite OPENAI_API_KEY in os.environ after normalization (no-op if missing)."""
    raw = os.environ.get("OPENAI_API_KEY", "")
    clean = normalize_api_key(raw)
    if clean and clean != raw:
        os.environ["OPENAI_API_KEY"] = clean


def openai_api_key_configured() -> bool:
    load_dotenv_files()
    sync_normalized_api_key_to_environ()
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def openai_env_warnings() -> list[str]:
    """
    Detect common Groq vs OpenAI misconfiguration (returns user-facing strings).
    """
    load_dotenv_files()
    sync_normalized_api_key_to_environ()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "").strip().lower()
    out: list[str] = []
    if not key:
        return out
    if key.startswith("gsk_"):
        if not base or "groq.com" not in base:
            out.append(
                "Your key looks like **Groq** (`gsk_...`). Set "
                "`OPENAI_BASE_URL=https://api.groq.com/openai/v1` (and a Groq model name)."
            )
    elif key.startswith("sk-") or key.startswith("sk_"):
        if "groq.com" in base:
            out.append(
                "You are using an **OpenAI**-style key with a **Groq** base URL. "
                "Use a Groq `gsk_...` key on Groq, or remove `OPENAI_BASE_URL` to use OpenAI."
            )
    return out
