"""
Student Name: Robert George Owoo
Index Number: 10022300108

Prompt engineering:
- Context injection
- Hallucination control
- Context window management (word budget)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PromptBuildResult:
    selected_context: list[dict]
    prompt: str


def select_context_by_word_budget(
    hits: Iterable[dict],
    *,
    max_words: int,
) -> list[dict]:
    selected: list[dict] = []
    used = 0
    for h in hits:
        text = str(h.get("text", ""))
        w = len(text.split())
        if w == 0:
            continue
        if used + w > max_words and selected:
            break
        selected.append(h)
        used += w
        if used >= max_words:
            break
    return selected


def build_prompt(
    *,
    user_query: str,
    retrieved_chunks: list[dict],
    max_context_words: int,
    prompt_version: str = "v2",
) -> PromptBuildResult:
    """
    prompt_version:
    - v1: simple context injection (concise)
    - v2: strict grounding + concise answers (default)
    - v2-verbose: same rules, longer answers (for prompt experiments)
    """
    selected = select_context_by_word_budget(retrieved_chunks, max_words=max_context_words)

    context_blocks: list[str] = []
    for i, c in enumerate(selected, start=1):
        header = f"[CONTEXT {i}] source={c.get('source')} chunk_id={c.get('chunk_id')} score={c.get('score'):.4f}"
        context_blocks.append(header + "\n" + c.get("text", ""))
    context_str = "\n\n".join(context_blocks).strip()

    if prompt_version == "v1":
        prompt = f"""You are an Academic City AI assistant.
Answer using ONLY the provided context. If the answer is not in the context, say: I don't know.

Style: at most 3 short sentences. Start with the direct answer. No meta commentary (do not say you read context, looked through blocks, etc.).

User question:
{user_query}

Context:
{context_str}

Answer (then one line: Sources: [n], ... only for blocks you used):"""
    elif prompt_version == "v2-verbose":
        prompt = f"""You are an Academic City AI assistant for question-answering over two datasets:
1) Ghana election results (CSV records)
2) MOFEP 2025 Budget Statement (PDF text)

RULES:
- Use ONLY the context blocks. Do not use outside knowledge.
- If the context is insufficient, say: "I don't have enough information in the provided documents."
- Be precise with numbers and names; do not guess.
- Provide citations by listing the context block numbers you used, like: Sources: [1], [3]

User question:
{user_query}

Context blocks:
{context_str}

Write the answer in 4-10 sentences. End with: Sources: [x], [y]"""
    else:
        # v2 (default): concise, grounded answers
        prompt = f"""You are an Academic City AI assistant over two datasets:
1) Ghana election results (CSV records)
2) MOFEP 2025 Budget Statement (PDF text)

RULES:
- Use ONLY the context blocks. No outside knowledge.
- If the context is insufficient, reply exactly: I don't have enough information in the provided documents.
- Be precise with numbers and names; do not guess.

STYLE (strict):
- At most 3 short sentences total.
- Sentence 1 must be the direct answer only (no preamble).
- Do not write phrases like: "based on the context", "I looked through", "the provided context blocks", "in conclusion".
- Last line only: Sources: [n] (cite only context indices you used; one block is fine as Sources: [1]).

User question:
{user_query}

Context blocks:
{context_str}

Answer:"""

    return PromptBuildResult(selected_context=selected, prompt=prompt)

