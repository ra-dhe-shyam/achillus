import json
import unittest
from unittest.mock import patch

import memory_ranking


class SemanticMemoryTests(unittest.TestCase):
    def setUp(self):
        self.memories = [
            {
                "id": 1,
                "category": "fact",
                "content": "My primary purpose is to improve through experience.",
                "created_at": 1000,
                "confidence": 1.0,
                "source": "explicit",
            },
            {
                "id": 2,
                "category": "fact",
                "content": "I enjoy cycling on weekends.",
                "created_at": 1000,
                "confidence": 1.0,
                "source": "explicit",
            },
        ]

    @patch("memory_ranking._cached_embeddings")
    @patch("memory_ranking._embed")
    @patch("memory_ranking.recall")
    def test_retrieves_related_meaning(
        self, mock_recall, mock_embed, mock_cached
    ):
        mock_recall.return_value = self.memories
        mock_embed.return_value = [1.0, 0.0]
        mock_cached.return_value = {
            1: [0.95, 0.05],
            2: [0.0, 1.0],
        }

        results = memory_ranking.rank_memories(
            "How can you become more capable by learning?",
            limit=5,
        )

        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["retrieval"], "semantic")
        self.assertGreater(results[0]["similarity"], 0.9)
        mock_cached.assert_called_once_with(self.memories)

    @patch("memory_ranking._embed", side_effect=RuntimeError("offline"))
    @patch("memory_ranking.recall")
    def test_falls_back_when_ollama_is_unavailable(
        self, mock_recall, mock_embed
    ):
        mock_recall.return_value = self.memories

        results = memory_ranking.rank_memories(
            "primary purpose improve experience",
            limit=5,
        )

        self.assertTrue(results)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["retrieval"], "keyword_fallback")

    @patch("memory_ranking.recall", return_value=[])
    def test_empty_memory_store_returns_empty_results(self, mock_recall):
        results = memory_ranking.rank_memories(
            "What do you remember?",
            limit=5,
        )

        self.assertEqual(results, [])

    @patch("memory_ranking.urlopen")
    def test_batch_embedding_uses_one_request_for_all_texts(self, mock_urlopen):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({
                    "embeddings": [[1.0, 0.0], [0.0, 1.0]]
                }).encode("utf-8")

        mock_urlopen.return_value = FakeResponse()
        vectors = memory_ranking._embed_many(["first text", "second text"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["input"], ["first text", "second text"])


if __name__ == "__main__":
    unittest.main()
