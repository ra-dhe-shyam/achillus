
"""Deterministic policy for selecting candidate memories."""

from dataclasses import dataclass
from core import remember

CATEGORIES = {
    "preference",
    "fact",
    "project",
    "temporary",
}

SOURCES = {
    "explicit",
    "inferred",
    "imported",
}


@dataclass(frozen=True)
class MemoryDecision:
    approved: bool
    category: str
    source: str
    reason: str


def evaluate_candidate(
    content,
    category,
    source="explicit",
    user_approved=False,
):
    """Evaluate a memory candidate without storing it."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Memory content must be non-empty text")

    if "\x00" in content:
        raise ValueError("Memory content cannot contain null bytes")

    if category not in CATEGORIES:
        raise ValueError("Unsupported memory category")

    if source not in SOURCES:
        raise ValueError("Unsupported memory source")

    if category == "temporary":
        return MemoryDecision(
            approved=False,
            category=category,
            source=source,
            reason="Temporary context is not retained by default",
        )

    if not user_approved:
        return MemoryDecision(
            approved=False,
            category=category,
            source=source,
            reason="User confirmation is required before storage",
        )

    if source == "inferred":
        return MemoryDecision(
            approved=False,
            category=category,
            source=source,
            reason="Inferred information requires user confirmation",
        )

    return MemoryDecision(
        approved=True,
        category=category,
        source=source,
        reason="Candidate passed policy checks",
    )

def remember_if_approved(
    content,
    category,
    source="explicit",
    user_approved=False,
    confidence=1.0,
):
    """Store a memory only when the policy approves it."""

    decision = evaluate_candidate(
        content=content,
        category=category,
        source=source,
        user_approved=user_approved,
    )

    if not decision.approved:
        return {
            "stored": False,
            "memory_id": None,
            "reason": decision.reason,
        }

    memory_id = remember(
        content=content,
        category=category,
        source=source,
        confidence=confidence,
    )

    return {
        "stored": True,
        "memory_id": memory_id,
        "reason": decision.reason,
    }
