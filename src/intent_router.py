"""Deterministic first-pass routing for explicit natural-language intents."""

import re


def classify_intent(message):
    """Return a high-confidence intent, otherwise preserve normal conversation."""
    if not isinstance(message, str):
        return {"intent": "conversation"}

    text = message.strip()
    normalized = re.sub(r"[.!?\s]+$", "", text.lower()).strip()

    if normalized in {
        "i'm done", "i'm done for now", "i am done",
        "that's all", "that is all", "stop the session",
        "end the session", "goodbye", "bye for now",
    }:
        return {"intent": "end_session"}

    match = re.match(
        r"^(?:please )?remember that\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        content = match.group(1).strip()
        if content:
            category = (
                "preference"
                if re.search(
                    r"\b(i prefer|i like|i dislike|i hate|i love|"
                    r"my preference is|i usually)\b",
                    content,
                    flags=re.IGNORECASE,
                )
                else "fact"
            )
            return {
                "intent": "remember",
                "content": content,
                "category": category,
            }

    return {"intent": "conversation"}
