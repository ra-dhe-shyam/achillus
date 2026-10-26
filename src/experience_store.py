
"""Privacy-conscious, append-only task experience ledger for Achillus.

Stores structured outcomes only when explicitly requested.
Does not store raw prompts, transcripts, or model responses by default.
"""

import json
import time
import uuid

from core import database_session


OUTCOMES = frozenset({"success", "failure", "partial", "corrected"})
SOURCES = frozenset({"user_feedback", "task_result", "evaluation"})


def initialize_experience_store():
    """Create the experience table idempotently."""
    with database_session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                experience_id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                outcome TEXT NOT NULL,
                source TEXT NOT NULL,
                summary TEXT NOT NULL,
                correction TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at REAL NOT NULL
            )
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiences_type_created
            ON experiences(task_type, created_at DESC)
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_experiences_outcome_created
            ON experiences(outcome, created_at DESC)
        """)


def _text(value, field, max_length):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")

    value = value.strip()

    if not value:
        raise ValueError(f"{field} must be non-empty")

    if "\x00" in value:
        raise ValueError(f"{field} cannot contain null bytes")

    if len(value) > max_length:
        raise ValueError(f"{field} exceeds {max_length} characters")

    return value


def record_experience(
    *,
    task_type,
    outcome,
    source,
    summary,
    correction=None,
    metadata=None,
    user_confirmed=False,
):
    """Persist one deliberately submitted task outcome.

    Explicit confirmation is required for every write.
    Pass concise summaries, not raw conversation transcripts.
    """
    if user_confirmed is not True:
        raise PermissionError(
            "Explicit confirmation is required to store an experience"
        )

    task_type = _text(task_type, "task_type", 80)
    summary = _text(summary, "summary", 1000)

    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        raise ValueError(
            f"outcome must be one of: {', '.join(sorted(OUTCOMES))}"
        )

    if not isinstance(source, str) or source not in SOURCES:
        raise ValueError(
            f"source must be one of: {', '.join(sorted(SOURCES))}"
        )

    if correction is not None:
        correction = _text(correction, "correction", 1000)

    if metadata is None:
        metadata = {}

    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")

    try:
        metadata_json = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metadata must contain JSON-serializable values"
        ) from exc

    if len(metadata_json) > 2000:
        raise ValueError("metadata exceeds 2000 serialized characters")

    experience_id = uuid.uuid4().hex
    created_at = time.time()

    initialize_experience_store()

    with database_session() as db:
        db.execute("""
            INSERT INTO experiences (
                experience_id,
                task_type,
                outcome,
                source,
                summary,
                correction,
                metadata_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experience_id,
            task_type,
            outcome,
            source,
            summary,
            correction,
            metadata_json,
            created_at,
        ))

    return {
        "experience_id": experience_id,
        "task_type": task_type,
        "outcome": outcome,
        "source": source,
        "summary": summary,
        "correction": correction,
        "metadata": metadata,
        "created_at": created_at,
    }


def list_experiences(*, task_type=None, outcome=None, limit=50):
    """Return recent structured experiences without raw conversation data."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 200
    ):
        raise ValueError("limit must be an integer between 1 and 200")

    if task_type is not None:
        task_type = _text(task_type, "task_type", 80)

    if outcome is not None and (
        not isinstance(outcome, str) or outcome not in OUTCOMES
    ):
        raise ValueError(
            f"outcome must be one of: {', '.join(sorted(OUTCOMES))}"
        )

    initialize_experience_store()

    sql = """
        SELECT experience_id, task_type, outcome, source, summary,
               correction, metadata_json, created_at
        FROM experiences
        WHERE 1 = 1
    """
    params = []

    if task_type is not None:
        sql += " AND task_type = ?"
        params.append(task_type)

    if outcome is not None:
        sql += " AND outcome = ?"
        params.append(outcome)

    sql += " ORDER BY created_at DESC, experience_id DESC LIMIT ?"
    params.append(limit)

    with database_session() as db:
        rows = db.execute(sql, params).fetchall()

    results = []

    for row in rows:
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        results.append(item)

    return results
