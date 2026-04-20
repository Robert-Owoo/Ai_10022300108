"""
Student Name: Robert George Owoo
Index Number: 10022300108

LLM adapter (OpenAI-compatible via `openai` Python SDK).
This file only handles generation; retrieval is separate.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from .env_bootstrap import load_dotenv_files, normalize_api_key, sync_normalized_api_key_to_environ


def _load_project_dotenv() -> None:
    load_dotenv_files()
    sync_normalized_api_key_to_environ()


@dataclass(frozen=True)
class LLMConfig:
    model: str
    api_key: str
    base_url: str | None = None


def load_llm_config_from_env() -> LLMConfig:
    _load_project_dotenv()
    api_key = normalize_api_key(os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Do one of the following:\n"
            "1) Create a file named `.env` in the project folder (next to `app.py`) containing:\n"
            "   OPENAI_API_KEY=your_key_here\n"
            "   (For Groq, also set OPENAI_BASE_URL and OPENAI_MODEL.)\n"
            "2) Or set the variable in PowerShell before starting Streamlit:\n"
            "   $env:OPENAI_API_KEY = \"your_key_here\""
        )
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    # Persist cleaned key for this process (helps debugging + avoids stray whitespace).
    os.environ["OPENAI_API_KEY"] = api_key
    return LLMConfig(model=model, api_key=api_key, base_url=base_url)


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    def generate(self, prompt: str, *, temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

