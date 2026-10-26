import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_interface import (
    HybridLanguageModel,
    LanguageModel,
    ModelError,
    MockLanguageModel,
    OllamaLanguageModel,
    OpenAICompatibleLanguageModel,
    build_language_model,
)


class FailingModel(LanguageModel):
    def generate(self, user_message, memories):
        raise ModelError("local unavailable")


class FixedModel(LanguageModel):
    def __init__(self, response):
        self.response = response

    def generate(self, user_message, memories):
        return self.response


class ModelInterfaceTests(unittest.TestCase):
    def test_mock_provider_remains_available(self):
        result = MockLanguageModel().generate("hello", [])
        self.assertIn("hello", result)

    def test_ollama_calls_local_chat_endpoint(self):
        provider = OllamaLanguageModel(model="test-model")
        with patch("model_interface._post_json", return_value={"message": {"content": "hello"}}) as post:
            self.assertEqual(provider.generate("hi", []), "hello")
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:11434/api/chat")
        self.assertFalse(post.call_args.args[1]["stream"])

    def test_hybrid_uses_local_first(self):
        local = FixedModel("local answer")
        cloud = OpenAICompatibleLanguageModel("test", "https://example.test/v1", "secret")
        model = HybridLanguageModel(local, cloud, allow_cloud_fallback=True)
        self.assertEqual(model.generate("hi", []), "local answer")

    def test_cloud_fallback_is_disabled_by_default(self):
        cloud = OpenAICompatibleLanguageModel("test", "https://example.test/v1", "secret")
        model = HybridLanguageModel(FailingModel(), cloud)
        with self.assertRaises(ModelError):
            model.generate("hi", [])

    def test_fallback_excludes_memories_by_default(self):
        cloud = OpenAICompatibleLanguageModel("test", "https://example.test/v1", "secret")
        model = HybridLanguageModel(FailingModel(), cloud, allow_cloud_fallback=True)
        with patch.object(cloud, "generate", return_value="cloud answer") as generate:
            self.assertEqual(model.generate("hi", [{"content": "private fact"}]), "cloud answer")
        self.assertFalse(generate.call_args.kwargs["include_memories"])

    def test_fallback_can_include_memories_only_when_enabled(self):
        cloud = OpenAICompatibleLanguageModel("test", "https://example.test/v1", "secret")
        model = HybridLanguageModel(
            FailingModel(), cloud,
            allow_cloud_fallback=True,
            allow_cloud_memory_context=True,
        )
        with patch.object(cloud, "generate", return_value="cloud answer") as generate:
            model.generate("hi", [{"content": "private fact"}])
        self.assertTrue(generate.call_args.kwargs["include_memories"])

    def test_build_hybrid_without_key_stays_local_only(self):
        with patch.dict("os.environ", {}, clear=True):
            model = build_language_model({"model_backend": "hybrid"})
        self.assertIsInstance(model, HybridLanguageModel)
        self.assertIsNone(model.cloud_model)
        self.assertFalse(model.allow_cloud_fallback)

    def test_invalid_backend_rejected(self):
        with self.assertRaises(ValueError):
            build_language_model({"model_backend": "unknown"})


if __name__ == "__main__":
    unittest.main()
