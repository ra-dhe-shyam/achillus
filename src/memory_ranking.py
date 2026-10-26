
"""Semantic memory retrieval using Ollama embeddings with keyword fallback."""

import json
import math
import os
import re
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from core import database_session, recall


EMBED_MODEL = os.environ.get("ACHILLUS_EMBED_MODEL", "all-minilm")
OLLAMA_URL = os.environ.get(
    "ACHILLUS_OLLAMA_URL", "http://127.0.0.1:11434"
).rstrip("/")
SIMILARITY_THRESHOLD = 0.20


def tokenize(text):
    """Extract lowercase alphanumeric terms for fallback retrieval."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _embed_many(texts):
    """Return embeddings for multiple texts in one Ollama request."""
    if not isinstance(texts, list) or not texts or not all(
        isinstance(text, str) and text.strip() for text in texts
    ):
        raise ValueError("texts must be a non-empty list of non-empty strings")

    payload = json.dumps({
        "model": EMBED_MODEL,
        "input": texts,
    }).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        vectors = data["embeddings"]
    except (URLError, TimeoutError, OSError, ValueError, KeyError,
            IndexError, TypeError) as exc:
        raise RuntimeError("Ollama embedding request failed") from exc

    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise RuntimeError("Ollama returned an unexpected number of embeddings")

    validated = []
    expected_dimensions = None
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Ollama returned an invalid embedding")
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in vector
        ):
            raise RuntimeError("Ollama returned non-finite embedding values")
        if expected_dimensions is None:
            expected_dimensions = len(vector)
        elif len(vector) != expected_dimensions:
            raise RuntimeError("Ollama returned inconsistent embedding dimensions")
        validated.append([float(value) for value in vector])

    return validated


def _embed(text):
    """Return one embedding, using the same validated batch interface."""
    return _embed_many([text])[0]


def _cosine_similarity(left, right):
    """Calculate cosine similarity between two vectors."""
    if len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot / (left_norm * right_norm)


def _cached_embeddings(memories):
    """Load cached vectors and batch-embed all missing or stale memories."""
    if not memories:
        return {}

    ids = [memory["id"] for memory in memories]
    placeholders = ",".join("?" for _ in ids)
    with database_session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS memory_embedding_cache (
                memory_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                PRIMARY KEY (memory_id, model)
            )
        """)
        rows = db.execute(
            f"""SELECT memory_id, content, vector_json
                FROM memory_embedding_cache
                WHERE model = ? AND memory_id IN ({placeholders})""",
            [EMBED_MODEL, *ids],
        ).fetchall()

    by_id = {row["memory_id"]: row for row in rows}
    vectors = {}
    missing = []
    for memory in memories:
        row = by_id.get(memory["id"])
        if row is not None and row["content"] == memory["content"]:
            try:
                vector = json.loads(row["vector_json"])
                if (isinstance(vector, list) and vector and all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in vector
                )):
                    vectors[memory["id"]] = [float(value) for value in vector]
                    continue
            except (ValueError, TypeError):
                pass
        missing.append(memory)

    if not missing:
        return vectors

    new_vectors = _embed_many([memory["content"] for memory in missing])
    with database_session() as db:
        db.executemany("""
            INSERT INTO memory_embedding_cache
                (memory_id, content, model, vector_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_id, model) DO UPDATE SET
                content = excluded.content,
                vector_json = excluded.vector_json
        """, [
            (memory["id"], memory["content"], EMBED_MODEL, json.dumps(vector))
            for memory, vector in zip(missing, new_vectors)
        ])

    for memory, vector in zip(missing, new_vectors):
        vectors[memory["id"]] = vector
    return vectors


def _keyword_rank(query, candidates, limit):
    """Fallback ranking when semantic retrieval is unavailable."""
    query_terms = tokenize(query)
    now = time.time()
    ranked = []

    for memory in candidates:
        content_terms = tokenize(memory["content"])
        if not query_terms or not content_terms:
            continue

        overlap = len(query_terms & content_terms) / len(query_terms)
        if overlap <= 0:
            continue

        confidence = float(memory["confidence"])
        age_days = max(0.0, (now - memory["created_at"]) / 86400.0)
        recency = math.exp(-age_days / 365.0)

        ranked.append({
            **memory,
            "relevance": round(
                0.75 * overlap + 0.15 * confidence + 0.10 * recency,
                4,
            ),
            "retrieval": "keyword_fallback",
        })

    ranked.sort(
        key=lambda item: (
            item["relevance"], item["created_at"], item["id"]
        ),
        reverse=True,
    )
    return ranked[:limit]


def rank_memories(query, category=None, limit=5):
    """Rank stored memories semantically, with keyword fallback."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be non-empty text")

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 100
    ):
        raise ValueError("Limit must be an integer between 1 and 100")

    query = query.strip()
    if not tokenize(query):
        raise ValueError("Query must contain searchable terms")

    candidates = recall(query="", category=category, limit=100)
    if not candidates:
        return []

    try:
        query_vector = _embed(query)
        ranked = []
        now = time.time()

        memory_vectors = _cached_embeddings(candidates)
        for memory in candidates:
            memory_vector = memory_vectors[memory["id"]]
            similarity = _cosine_similarity(query_vector, memory_vector)

            if similarity < SIMILARITY_THRESHOLD:
                continue

            age_days = max(
                0.0, (now - memory["created_at"]) / 86400.0
            )
            recency = math.exp(-age_days / 365.0)
            confidence = float(memory["confidence"])

            score = (
                0.80 * similarity
                + 0.15 * confidence
                + 0.05 * recency
            )

            ranked.append({
                **memory,
                "relevance": round(score, 4),
                "similarity": round(similarity, 4),
                "retrieval": "semantic",
            })

        ranked.sort(
            key=lambda item: (
                item["relevance"], item["created_at"], item["id"]
            ),
            reverse=True,
        )
        return ranked[:limit]

    except (RuntimeError, OSError, ValueError, TypeError):
        return _keyword_rank(query, candidates, limit)
