import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import core
from model_interface import MockLanguageModel


class ConversationRuntimeTests(unittest.TestCase):
    def test_chat_loop_responds_and_exits(self):
        messages = iter(["hello Achillus", "/quit"])
        output = []
        config = {"conversation_memory_limit": 5, "model_backend": "mock"}
        with patch("conversation.rank_memories", return_value=[]):
            core.run_chat(config, input_fn=lambda prompt: next(messages), output_fn=output.append)
        self.assertTrue(any(item.startswith("Achillus: Development reply") for item in output))
        self.assertTrue(any("session ended" in item for item in output))

    def test_empty_messages_are_ignored(self):
        messages = iter(["   ", "/quit"])
        output = []
        core.run_chat({"model_backend": "mock"}, input_fn=lambda prompt: next(messages), output_fn=output.append)
        self.assertFalse(any(item.startswith("Achillus: Development reply") for item in output))

    def test_response_records_phase_timings(self):
        from conversation import ConversationEngine

        engine = ConversationEngine(model=MockLanguageModel())
        with patch("conversation.rank_memories", return_value=[]):
            reply = engine.respond("hello")

        self.assertTrue(reply)
        self.assertEqual(
            set(engine.last_timings),
            {"memory_retrieval_ms", "cognition_and_generation_ms", "total_ms"},
        )
        self.assertGreaterEqual(engine.last_timings["total_ms"], 0)
        self.assertGreaterEqual(engine.last_timings["memory_retrieval_ms"], 0)
        self.assertGreaterEqual(engine.last_timings["cognition_and_generation_ms"], 0)


if __name__ == "__main__":
    unittest.main()
