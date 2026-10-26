import math
import numbers
import json
import logging
import sqlite3
import time
import argparse
from pathlib import Path
from contextlib import contextmanager

def validate_text(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    value = value.strip()
    if not value:
        raise ValueError(f"{field} cannot be empty")
    if "\x00" in value:
        raise ValueError(f"{field} cannot contain null bytes")
    return value


def validate_confidence(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError("Confidence must be finite and between 0 and 1")
    return float(value)


def validate_memory_id(memory_id):
    if (
        isinstance(memory_id, bool)
        or not isinstance(memory_id, int)
        or memory_id < 1
    ):
        raise ValueError("Memory ID must be a positive integer")
    return memory_id


def validate_source(source):
    if source not in {"explicit", "inferred", "imported"}:
        raise ValueError("Invalid memory source")
    return source

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "settings.json"
DATABASE = ROOT / "data" / "memory.db"
LOGS = ROOT / "logs"


def load_config():
    with CONFIG.open(encoding="utf-8") as file:
        return json.load(file)


def setup_logging(level="INFO"):
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS / "core.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger("personal_ai")


def connect_db():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DATABASE, timeout=5)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    return db



@contextmanager
def database_session():
    db = connect_db()
    try:
        with db:
            yield db
    finally:
        db.close()


def initialize_database():
    with database_session() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'explicit'
            )
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memory_category
            ON memory(category)
        """)


def remember(content, category="fact", confidence=1.0,
             source="explicit"):
    content = validate_text(content, "Memory content")
    category = validate_text(category, "Memory category")
    confidence = validate_confidence(confidence)
    source = validate_source(source)

    with database_session() as db:
        cursor = db.execute("""
            INSERT INTO memory
                (category, content, created_at, confidence, source)
            VALUES (?, ?, ?, ?, ?)
        """, (category, content, time.time(), confidence, source))
        return cursor.lastrowid



def recall(query="", category=None, limit=20):
    if not isinstance(query, str):
        raise ValueError("Search query must be text")
    if "\x00" in query:
        raise ValueError("Search query cannot contain null bytes")

    query = query.strip()

    if category is not None:
        category = validate_text(category, "Memory category")

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= 100
    ):
        raise ValueError("Limit must be an integer between 1 and 100")

    sql = """
        SELECT id, category, content, created_at, confidence, source
        FROM memory
        WHERE content LIKE ? ESCAPE '\\'
    """
    escaped = (
        query.replace("\\", "\\\\")
             .replace("%", "\\%")
             .replace("_", "\\_")
    )
    params = [f"%{escaped}%"]

    if category is not None:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(limit)

    with database_session() as db:
        return [
            dict(row)
            for row in db.execute(sql, params).fetchall()
        ]



def update_memory(memory_id, content=None, category=None,
                  confidence=None, source=None):
    memory_id = validate_memory_id(memory_id)
    fields = []
    values = []

    if content is not None:
        fields.append("content = ?")
        values.append(validate_text(content, "Memory content"))

    if category is not None:
        fields.append("category = ?")
        values.append(validate_text(category, "Memory category"))

    if confidence is not None:
        fields.append("confidence = ?")
        values.append(validate_confidence(confidence))

    if source is not None:
        fields.append("source = ?")
        values.append(validate_source(source))

    if not fields:
        raise ValueError("No changes supplied")

    values.append(memory_id)
    with database_session() as db:
        cursor = db.execute(
            f"UPDATE memory SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        return cursor.rowcount == 1



def forget(memory_id):
    memory_id = validate_memory_id(memory_id)
    with database_session() as db:
        cursor = db.execute(
            "DELETE FROM memory WHERE id = ?", (memory_id,)
        )
        return cursor.rowcount == 1


def health_check():
    with database_session() as db:
        db.execute("SELECT 1").fetchone()
    return {"status": "healthy", "database": "available"}


def run_chat(config, input_fn=input, output_fn=print):
    """Run the interactive text conversation loop."""
    from conversation import ConversationEngine
    from model_interface import ModelError, build_language_model

    engine = ConversationEngine(
        memory_limit=config.get("conversation_memory_limit", 5),
        model=build_language_model(config),
    )
    output_fn("Achillus is ready. Type /quit to exit.")
    while True:
        try:
            message = input_fn("You: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nAchillus session ended.")
            break

        if not isinstance(message, str):
            output_fn("Please enter text.")
            continue

        command = message.strip()

        if command.lower() in {"/quit", "/exit"}:
            output_fn("Achillus session ended.")
            break
        if command == "/feedback" or command.startswith("/feedback "):
            from feedback import handle_feedback_command
            output_fn(handle_feedback_command(command))
            continue
        if command.lower() == "/voice":
            from audio_runtime import AudioRuntime, AudioUnavailable
            try:
                voice = AudioRuntime(config)
                result = voice.converse_once(engine)
                if not result["transcript"]:
                    output_fn("No speech recognized.")
                    continue
                output_fn(f"You (voice): {result['transcript']}")
                output_fn(f"Achillus: {result['reply']}")
            except AudioUnavailable as exc:
                output_fn(f"Voice unavailable: {exc}")
            continue

        if command.lower() == "/voice-loop":
            from audio_runtime import AudioRuntime, AudioUnavailable
            try:
                voice = AudioRuntime(config)
                voice.converse_hands_free(engine, output_fn=output_fn)
            except KeyboardInterrupt:
                output_fn("Hands-free mode stopped.")
            except AudioUnavailable as exc:
                output_fn(f"Voice unavailable: {exc}")
            continue

        if command.startswith("/remember "):
            content = command[len("/remember "):].strip()

            if not content:
                output_fn("Usage: /remember <text>")
                continue

            try:
                memory_id = remember(
                    content,
                    category="fact",
                    source="explicit",
                )
                output_fn(f"Memory saved (ID: {memory_id}).")
            except (ValueError, TypeError) as exc:
                output_fn(f"Could not save memory: {exc}")

            continue

        if not command:
            continue

        try:
            reply = engine.respond(command)
        except (ValueError, ModelError, RuntimeError) as exc:
            # Avoid logging or echoing prompts, memories, API keys, or full payloads.
            output_fn(f"Achillus could not generate a response: {exc}")
            continue

        output_fn(f"Achillus: {reply}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Achillus personal AI")
    parser.add_argument(
        "--health-only", action="store_true",
        help="initialize the foundation and print health status without chatting",
    )
    args = parser.parse_args(argv)

    config = load_config()
    logger = setup_logging(config["log_level"])
    initialize_database()
    logger.info("%s v%s starting", config["service_name"], config["version"])
    logger.info("Raw audio retention: %s", config["store_raw_audio"])
    logger.info("Health check: %s", health_check())
    if args.health_only:
        return
    run_chat(config)


if __name__ == "__main__":
    main()
