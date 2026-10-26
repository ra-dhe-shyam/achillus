import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_tools import AgentTools, ToolError


class AgentToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "hello.txt").write_text("hello Achillus", encoding="utf-8")
        self.output = []
        self.tools = AgentTools(self.root, approval_input=lambda prompt: "", output=self.output.append,
                                ollama_url="http://127.0.0.1:1")

    def tearDown(self):
        self.temp.cleanup()

    def test_list_files(self):
        result = self.tools.list_files({"path": "src"})
        self.assertEqual(result["entries"][0]["name"], "hello.txt")

    def test_read_file(self):
        self.assertEqual(self.tools.read_file({"path": "src/hello.txt"})["content"], "hello Achillus")

    def test_parent_traversal_denied(self):
        with self.assertRaises(ToolError):
            self.tools.read_file({"path": "../outside.txt"})

    def test_absolute_path_denied(self):
        with self.assertRaises(ToolError):
            self.tools.read_file({"path": str(self.root / "src" / "hello.txt")})

    def test_secret_and_database_files_denied(self):
        (self.root / ".env").write_text("secret")
        (self.root / "x.db").write_text("database")
        for name in (".env", "x.db"):
            with self.assertRaises(ToolError):
                self.tools.read_file({"path": name})

    def test_script_creation_does_not_execute(self):
        result = self.tools.create_python_script({"filename": "sample.py", "code": "print('ok')\n"})
        self.assertEqual(result["status"], "created_not_executed")
        self.assertTrue((self.root / result["path"]).is_file())
        self.assertTrue(any("SCRIPT FOR REVIEW" in line for line in self.output))

    def test_script_execution_requires_exact_terminal_confirmation(self):
        created = self.tools.create_python_script({"filename": "sample.py", "code": "print('ok')\n"})
        result = self.tools.run_python_script({"path": created["path"], "sha256": created["sha256"]})
        self.assertEqual(result["status"], "denied")

    def test_modified_script_hash_denied(self):
        created = self.tools.create_python_script({"filename": "sample.py", "code": "print('ok')\n"})
        (self.root / created["path"]).write_text("print('changed')", encoding="utf-8")
        with self.assertRaises(ToolError):
            self.tools.run_python_script({"path": created["path"], "sha256": created["sha256"]})

    def test_execution_requires_review_phrase_and_runs_bounded_process(self):
        created = self.tools.create_python_script({"filename": "sample.py", "code": "print('ok')\n"})
        self.tools.approval_input = lambda prompt: "RUN PYTHON " + created["sha256"]
        result = self.tools.run_python_script({"path": created["path"], "sha256": created["sha256"]})
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["stdout"].strip(), "ok")


if __name__ == "__main__":
    unittest.main()
