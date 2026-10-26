
"""Conversation context and response orchestration."""

import logging
import time

from memory_ranking import rank_memories
from model_interface import LanguageModel, MockLanguageModel
from cognition import CognitiveLoop

logger = logging.getLogger("personal_ai")

class ConversationEngine:

    def __init__(self, memory_limit=5, model=None):
        if (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or not 1 <= memory_limit <= 20
        ):
            raise ValueError("memory_limit must be an integer from 1 to 20")

        if model is not None and not isinstance(model, LanguageModel):
            raise TypeError("model must implement LanguageModel")

        self.memory_limit = memory_limit
        self.model = model or MockLanguageModel()
        self.cognition = CognitiveLoop(self.model)
        self.history = []
        self.history_limit = 20
    def prepare_context(self, user_message):
        if not isinstance(user_message, str):
            raise ValueError("Message must be text")

        user_message = user_message.strip()

        if not user_message:
            raise ValueError("Message cannot be empty")

        if "\x00" in user_message:
            raise ValueError("Message cannot contain null bytes")

        retrieval_started = time.perf_counter()
        memories = rank_memories(
            user_message,
            limit=self.memory_limit,
        )
        self._last_retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        return {
            "user_message": user_message,
            "memories": [
                {
                    "category": memory["category"],
                    "content": memory["content"],
                    "confidence": memory["confidence"],
                    "source": memory["source"],
                }
                for memory in memories
            ],
        }

    def respond(self, user_message):
        started = time.perf_counter()
        context = self.prepare_context(user_message)
        retrieval_ms = getattr(self, "_last_retrieval_ms", 0.0)

        generation_started = time.perf_counter()
        reply = self.cognition.run(
            context["user_message"],
            context["memories"],
            history=self.history,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000

        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Model returned an empty or invalid reply")

        reply = reply.strip()
        self.history.extend([
            {"role": "user", "content": context["user_message"]},
            {"role": "assistant", "content": reply},
        ])
        self.history = self.history[-self.history_limit:]
    def respond(self, user_message):
        started = time.perf_counter()
        context = self.prepare_context(user_message)
        retrieval_ms = getattr(self, "_last_retrieval_ms", 0.0)

        generation_started = time.perf_counter()
        reply = self.cognition.run(
            context["user_message"],
            context["memories"],
            history=self.history,
        )
        generation_ms = (time.perf_counter() - generation_started) * 1000

        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Model returned an empty or invalid reply")

        reply = reply.strip()
        self.history.extend([
            {"role": "user", "content": context["user_message"]},
            {"role": "assistant", "content": reply},
        ])
        self.history = self.history[-self.history_limit:]

        self.last_timings = {
            "memory_retrieval_ms": round(retrieval_ms, 2),
            "cognition_and_generation_ms": round(generation_ms, 2),
            "total_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        logger.info(
            "Conversation timings: memory_retrieval_ms=%.2f "
            "cognition_and_generation_ms=%.2f total_ms=%.2f",
            self.last_timings["memory_retrieval_ms"],
            self.last_timings["cognition_and_generation_ms"],
            self.last_timings["total_ms"],
        )
        return reply
