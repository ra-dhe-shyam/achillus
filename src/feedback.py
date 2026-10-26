
"""Explicit user-feedback commands for the Achillus experience ledger."""

import json
import logging

from experience_store import record_experience

logger = logging.getLogger("personal_ai.feedback")

COMMAND = "/feedback"

HELP = """Usage:
  /feedback {"task_type":"explanation","outcome":"corrected","summary":"The explanation confused dipole moment with polarizability","correction":"Explain induced dipole as a response to an external electric field"}

Allowed outcomes: success, failure, partial, corrected
Allowed sources: user_feedback, task_result, evaluation

Only explicit /feedback commands create experience records.
"""


def handle_feedback_command(message):
    """Handle an explicit /feedback command; return None for other messages."""

    if not isinstance(message, str):
        return None

    command = message.strip()

    if not command.startswith(COMMAND):
        return None

    if command != COMMAND and not command.startswith(COMMAND + " "):
        return None

    payload_text = command[len(COMMAND):].strip()

    if not payload_text:
        return HELP

    try:
        payload = json.loads(payload_text)
    except (json.JSONDecodeError, TypeError):
        return "Invalid feedback JSON.\n\n" + HELP

    if not isinstance(payload, dict):
        return "Feedback must be a JSON object."

    allowed_fields = {
        "task_type",
        "outcome",
        "source",
        "summary",
        "correction",
        "metadata",
    }

    unexpected = set(payload) - allowed_fields
    if unexpected:
        return "Unsupported field(s): " + ", ".join(sorted(unexpected))

    required_fields = {"task_type", "outcome", "summary"}
    missing = required_fields - set(payload)

    if missing:
        return "Missing required field(s): " + ", ".join(sorted(missing))

    try:
        record_experience(
            task_type=payload["task_type"],
            outcome=payload["outcome"],
            source=payload.get("source", "user_feedback"),
            summary=payload["summary"],
            correction=payload.get("correction"),
            metadata=payload.get("metadata"),
            user_confirmed=True,
        )
    except (ValueError, TypeError) as exc:
        return f"Feedback not recorded: {exc}"
    except Exception:
        logger.exception("Failed to record explicit user feedback")
        return "Feedback could not be recorded because of an internal error."

    return "Feedback recorded in the experience ledger."
