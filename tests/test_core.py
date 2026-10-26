	
import sys
import tempfile
import unittest
import sqlite3
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import core

from contextlib import closing
from memory_policy import evaluate_candidate, remember_if_approved
from memory_ranking import rank_memories
from conversation import ConversationEngine
from model_interface import MockLanguageModel

class CoreTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db = Path(self.temp_dir.name) / "memory.db"
        self.db_patch = patch.object(core, "DATABASE", self.test_db)
        self.db_patch.start()
        core.initialize_database()

    def tearDown(self):
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_config(self):
        config = core.load_config()
        self.assertTrue(config["offline_first"])
        self.assertFalse(config["store_raw_audio"])

    def test_database_initialization(self):
        with closing(sqlite3.connect(self.test_db)) as db:
            with db:
                tables = {
                    row[0] for row in db.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table'"
                    )
                }

        self.assertIn("memory", tables)

    def test_health_check(self):
        result = core.health_check()
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["database"], "available")

    def test_remember_and_recall(self):
        memory_id = core.remember(
            "Prefers offline-first AI", category="preference"
        )
        results = core.recall(category="preference")
        self.assertTrue(any(
            item["id"] == memory_id
            and item["content"] == "Prefers offline-first AI"
            for item in results
        ))

    def test_update_memory(self):
        memory_id = core.remember("Old value")
        self.assertTrue(core.update_memory(
            memory_id, content="Updated value"
        ))
        results = core.recall(query="Updated value")
        self.assertTrue(any(
            item["id"] == memory_id for item in results
        ))

    def test_forget_memory(self):
        memory_id = core.remember("Temporary memory")
        self.assertTrue(core.forget(memory_id))
        self.assertFalse(core.forget(memory_id))
        self.assertFalse(any(
            item["id"] == memory_id for item in core.recall()
        ))

    def test_category_filter(self):
        core.remember("A preference", category="preference")
        core.remember("A fact", category="fact")
        results = core.recall(category="preference")
        self.assertTrue(results)
        self.assertTrue(all(
            item["category"] == "preference" for item in results
        ))

    def test_memory_persistence(self):
        memory_id = core.remember("Persistence marker")
        results = core.recall(query="Persistence marker")
        self.assertTrue(any(
            item["id"] == memory_id for item in results
        ))
   
    def test_invalid_memory_content(self):
        for value in ("", "   ", None, 123, "bad\x00text"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    core.remember(value)

    def test_invalid_confidence(self):
        for value in (-0.1, 1.1, True, float("nan"),
                      float("inf"), "0.5"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(ValueError):
                    core.remember("Test memory", confidence=value)

    def test_invalid_source(self):
        with self.assertRaises(ValueError):
            core.remember("Test memory", source="unknown")

    def test_invalid_memory_ids(self):
        for memory_id in (0, -1, True, 1.5, "1", None):
            with self.subTest(memory_id=repr(memory_id)):
                with self.assertRaises(ValueError):
                    core.forget(memory_id)

    def test_invalid_recall_limits(self):
        for limit in (0, -1, 101, True, 2.5, "10"):
            with self.subTest(limit=repr(limit)):
                with self.assertRaises(ValueError):
                    core.recall(limit=limit)

    def test_invalid_recall_query(self):
        for query in (None, 123, "bad\x00query"):
            with self.subTest(query=repr(query)):
                with self.assertRaises(ValueError):
                    core.recall(query=query)

    def test_invalid_update(self):
        memory_id = core.remember("Valid memory")

        with self.assertRaises(ValueError):
            core.update_memory(memory_id, content="   ")

        with self.assertRaises(ValueError):
            core.update_memory(memory_id, confidence=float("nan"))

        with self.assertRaises(ValueError):
            core.update_memory(memory_id, source="unknown")

        with self.assertRaises(ValueError):
            core.update_memory(memory_id)
   
    def test_failed_update_preserves_memory(self):
        memory_id = core.remember("Keep this exact text")

        with self.assertRaises(ValueError):
            core.update_memory(
                memory_id,
                content="Corrupted replacement",
                confidence=float("nan"),
            )

        results = core.recall(query="Keep this exact text")
        self.assertTrue(any(
            item["id"] == memory_id
            and item["content"] == "Keep this exact text"
            for item in results
        ))

    def test_search_wildcards_are_literal(self):
        core.remember("100% complete")
        core.remember("1000 complete")
        core.remember("file_name")
        core.remember("fileXname")

        percent_results = core.recall(query="%")
        underscore_results = core.recall(query="_")

        self.assertTrue(all(
            "%" in item["content"] for item in percent_results
        ))
        self.assertTrue(all(
            "_" in item["content"] for item in underscore_results
        ))
   
    def test_policy_requires_user_approval(self):
        result = evaluate_candidate(
            "Prefers offline-first AI",
            "preference",
            user_approved=False,
        )
        self.assertFalse(result.approved)

    def test_policy_accepts_confirmed_preference(self):
        result = evaluate_candidate(
            "Prefers offline-first AI",
            "preference",
            source="explicit",
            user_approved=True,
        )
        self.assertTrue(result.approved)

    def test_policy_rejects_inference(self):
        result = evaluate_candidate(
            "May prefer offline-first AI",
            "preference",
            source="inferred",
            user_approved=True,
        )
        self.assertFalse(result.approved)

    def test_policy_rejects_temporary_context(self):
        result = evaluate_candidate(
            "Needs a reminder today",
            "temporary",
            user_approved=True,
        )
        self.assertFalse(result.approved)

    def test_policy_rejects_invalid_category(self):
        with self.assertRaises(ValueError):
            evaluate_candidate(
                "Some information",
                "unknown",
                user_approved=True,
            )

    def test_policy_rejects_empty_content(self):
        with self.assertRaises(ValueError):
            evaluate_candidate(
                "   ",
                "fact",
                user_approved=True,
            )
   
    def test_gate_does_not_store_unapproved_memory(self):
        result = remember_if_approved(
            "Unapproved preference",
            "preference",
            user_approved=False,
        )
        self.assertFalse(result["stored"])
        self.assertIsNone(result["memory_id"])
        self.assertFalse(core.recall(query="Unapproved preference"))

    def test_gate_stores_approved_memory(self):
        result = remember_if_approved(
            "Confirmed preference",
            "preference",
            source="explicit",
            user_approved=True,
        )
        self.assertTrue(result["stored"])
        self.assertIsNotNone(result["memory_id"])
        self.assertTrue(core.recall(query="Confirmed preference"))

    def test_gate_rejects_inferred_memory(self):
        result = remember_if_approved(
            "Possible preference",
            "preference",
            source="inferred",
            user_approved=True,
        )
        self.assertFalse(result["stored"])
        self.assertFalse(core.recall(query="Possible preference"))
   
    def test_ranking_prefers_matching_terms(self):
        core.remember(
            "Raspberry Pi offline AI project",
            category="project",
            confidence=1.0,
        )
        core.remember(
            "Cooking recipes and ingredients",
            category="fact",
            confidence=1.0,
        )

        results = rank_memories("Raspberry Pi AI")
        self.assertTrue(results)
        self.assertIn("Raspberry Pi", results[0]["content"])
        self.assertIn("relevance", results[0])

    def test_ranking_confidence_affects_score(self):
        core.remember(
            "Raspberry Pi project",
            confidence=1.0,
        )
        core.remember(
            "Raspberry Pi project",
            confidence=0.2,
        )

        results = rank_memories("Raspberry Pi")
        self.assertEqual(len(results), 2)
        self.assertGreater(
            results[0]["relevance"],
            results[1]["relevance"],
        )

    def test_ranking_category_filter(self):
        core.remember(
            "Raspberry Pi project",
            category="project",
        )
        core.remember(
            "Raspberry Pi preference",
            category="preference",
        )

        results = rank_memories("Raspberry Pi", category="project")
        self.assertTrue(results)
        self.assertTrue(all(
            item["category"] == "project" for item in results
        ))

    def test_ranking_rejects_invalid_query(self):
        for query in ("", "   ", None, 123, "!!!"):
            with self.subTest(query=repr(query)):
                with self.assertRaises(ValueError):
                    rank_memories(query)

   
    def test_conversation_prepares_relevant_context(self):
        core.remember(
            "The personal AI runs on Raspberry Pi",
            category="project",
        )
        engine = ConversationEngine()
        result = engine.prepare_context(
            "Tell me about the Raspberry Pi project"
        )

        self.assertEqual(
            result["user_message"],
            "Tell me about the Raspberry Pi project",
        )
        self.assertTrue(result["memories"])
        self.assertTrue(any(
            "Raspberry Pi" in item["content"]
            for item in result["memories"]
        ))

    def test_conversation_rejects_invalid_messages(self):
        engine = ConversationEngine()

        for message in ("", "   ", None, 123, "bad\x00message"):
            with self.subTest(message=repr(message)):
                with self.assertRaises(ValueError):
                    engine.prepare_context(message)

    def test_conversation_validates_memory_limit(self):
        for limit in (0, -1, 21, True, 2.5, "5"):
            with self.subTest(limit=repr(limit)):
                with self.assertRaises(ValueError):
                    ConversationEngine(memory_limit=limit)
   
    def test_conversation_generates_reply(self):
        engine = ConversationEngine(model=MockLanguageModel())
        reply = engine.respond("Hello, assistant")
        self.assertIn("Hello, assistant", reply)
        self.assertIn("Development reply", reply)

    def test_model_rejects_invalid_input(self):
        model = MockLanguageModel()
        with self.assertRaises(ValueError):
            model.generate("", [])
        with self.assertRaises(ValueError):
            model.generate("Hello", None)


if __name__ == "__main__":
    unittest.main()
	
