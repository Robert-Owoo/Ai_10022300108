"""
Student Name: Robert George Owoo
Index Number: 10022300108

Detect bare greetings so the UI can reply politely without calling the LLM.
"""

from __future__ import annotations

import re
from datetime import datetime


_TASK_WORDS = re.compile(
    r"\b(help|need|want|show|tell|explain|find|list|what|when|where|why|"
    r"how\s+many|how\s+did|vote|election|budget|region|csv|pdf|ghana|"
    r"project|question|data|result|row|party|candidate|year)\b",
    re.IGNORECASE,
)


def is_greeting_only(text: str) -> bool:
    """
    True when the message looks like a simple greeting with no substantive request.
    Examples: "hello", "hi", "hey", "good morning", "good evening!"
    """
    raw = text.strip()
    if not raw:
        return False
    if _TASK_WORDS.search(raw):
        return False

    t = raw.lower()
    t = re.sub(r"[!?.…]+$", "", t).strip()
    words = t.split()
    if len(words) > 8:
        return False

    # Allow short add-ons: "hello there", "hi everyone"
    if re.fullmatch(
        r"(hello|hi|hey|hiya|howdy|yo|sup|greetings)(\s+(there|everyone|all|team|friend(s)?))?",
        t,
    ):
        return True
    if re.fullmatch(r"good\s+(morning|afternoon|evening|night)(!)?", t):
        return True
    if re.fullmatch(r"(morning|afternoon|evening)(!)?", t):
        return True
    if re.fullmatch(r"how\s+are\s+you\??", t):
        return True
    if t in {"gm", "ga", "ge"}:
        return False  # too ambiguous
    return False


def greeting_reply(user_text: str) -> str:
    """Short, professional reply; mirrors time-of-day when appropriate."""
    t = user_text.strip().lower()
    hour = datetime.now().hour
    if 5 <= hour < 12:
        default_tod = "Good morning"
    elif 12 <= hour < 17:
        default_tod = "Good afternoon"
    else:
        default_tod = "Good evening"

    if "good morning" in t or re.fullmatch(r"morning(!)?", t.strip().lower()):
        return (
            "Good morning! I'm your Academic City RAG assistant for the Ghana election CSV "
            "and the 2025 budget statement. How can I help you today?"
        )
    if "good afternoon" in t or re.fullmatch(r"afternoon(!)?", t.strip().lower()):
        return (
            "Good afternoon! How can I assist you with the election data or budget documents?"
        )
    if "good evening" in t or "good night" in t or re.fullmatch(r"evening(!)?", t.strip().lower()):
        return (
            "Good evening! What would you like to explore in the election results or 2025 budget?"
        )

    if t.startswith("hello") or t == "hello there":
        return (
            "Hello! How can I assist you with the Ghana election results or the 2025 budget statement today?"
        )
    if t.startswith("hi") or t == "hi there":
        return "Hi there! What would you like to know about the election CSV or the budget PDF?"
    if t.startswith("hey") or t.startswith("hiya") or t.startswith("howdy"):
        return (
            f"{default_tod}! I'm here for questions about the election dataset or the 2025 budget. "
            "How can I help?"
        )

    return f"{default_tod}! How may I help you today?"
