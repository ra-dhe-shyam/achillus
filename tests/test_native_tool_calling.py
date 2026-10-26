import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_interface import OllamaLanguageModel, ModelError
from cognition import CognitiveLoop
from model_interface import LanguageModel


class ScriptedToolModel(LanguageModel):
    def __init__(self): self.calls = 0
    def generate(self, user_message, memories): return "direct answer"
    def generate_with_tools(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {"type": "tool_call", "name": "list_files", "arguments": {"path": "."},
                    "assistant_message": {"role": "assistant", "content": "", "tool_calls": []}, "provider": "test"}
        return {"type": "final", "content": "I inspected the workspace."}
    def tool_result_message(self, tool_call, result):
        return {"role": "tool", "content": str(result)}


class NativeToolCallingTests(unittest.TestCase):
    def test_direct_answer_requires_one_generation(self):
        model = ScriptedToolModel()
        model.generate_with_tools = lambda messages, tools: {"type": "final", "content": "hello"}
        loop = CognitiveLoop(model, tool_registry={"list_files": lambda args: self.fail("tool should not run")})
        self.assertEqual(loop.run("say hello", []), "hello")

    def test_ollama_tool_call_parsed(self):
        provider = OllamaLanguageModel(model="test")
        response = {"message": {"content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "README.txt"}}}]}}
        with patch("model_interface._post_json", return_value=response):
            result = provider.generate_with_tools([{"role": "user", "content": "read README"}], [])
        self.assertEqual(result["type"], "tool_call")
        self.assertEqual(result["name"], "read_file")
        self.assertEqual(result["arguments"]["path"], "README.txt")

    def test_ollama_plain_response_is_final(self):
        provider = OllamaLanguageModel(model="test")
        with patch("model_interface._post_json", return_value={"message": {"content": "hello"}}):
            result = provider.generate_with_tools([{"role": "user", "content": "hi"}], [])
        self.assertEqual(result, {"type": "final", "content": "hello"})

    def test_invalid_tool_call_shape_rejected(self):
        provider = OllamaLanguageModel(model="test")
        with patch("model_interface._post_json", return_value={"message": {"content": "", "tool_calls": [{"function": {"name": "x", "arguments": "bad"}}]}}):
            with self.assertRaises(ModelError): provider.generate_with_tools([], [])

    def test_default_cognition_loop_finishes_after_a_tool_call(self):
        model = ScriptedToolModel()
        loop = CognitiveLoop(
            model,
            tool_registry={"list_files": lambda args: {"entries": ["README.txt"]}},
        )
        self.assertEqual(loop.run("inspect files", []), "I inspected the workspace.")
        self.assertEqual(model.calls, 2)

    def test_cognition_uses_tool_result_then_final(self):
        model = ScriptedToolModel()
        calls = []
        loop = CognitiveLoop(model, max_steps=2, tool_registry={"list_files": lambda args: calls.append(args) or {"entries": ["README.txt"]}})
        self.assertEqual(loop.run("inspect files", []), "I inspected the workspace.")
        self.assertEqual(model.calls, 2)
        self.assertEqual(calls, [{"path": "."}])


if __name__ == "__main__": unittest.main()
