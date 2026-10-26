"""Controlled project tools exposed to Achillus through the cognition runtime.

Filesystem access is restricted to the project root. Python execution is gated by
an explicit terminal confirmation of the exact script hash; it is not a security
sandbox and must only be used for code the user has inspected and trusts.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request

MAX_READ_BYTES = 64 * 1024
MAX_SCRIPT_BYTES = 20 * 1024
MAX_OUTPUT_CHARS = 8_000
PYTHON_TIMEOUT_SECONDS = 10
BLOCKED_PARTS = {".git", ".venv", "__pycache__", "data", "logs", "models", ".achillus-backups"}
BLOCKED_NAMES = {".env", ".env.local", "id_rsa", "id_ed25519"}
BLOCKED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key"}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List non-sensitive files and directories in the Achillus project workspace. Paths are relative to the project root.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Optional relative directory, default '.'"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file inside the Achillus project workspace. Does not read secrets, databases, virtual environments, or files outside the workspace.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path from project root"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "check_ollama",
        "description": "Check the configured local Ollama API and list available model names. Does not run shell commands or contact cloud services.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "create_python_script",
        "description": "Create a Python script draft in .achillus/scripts/pending for user review. Do not claim it has been executed.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string", "description": "Simple filename ending in .py"},
            "code": {"type": "string", "description": "Complete Python source code"}}, "required": ["filename", "code"]}}},
    {"type": "function", "function": {
        "name": "run_python_script",
        "description": "Run a reviewed script from .achillus/scripts/pending only after printing its exact source and receiving explicit terminal confirmation. Execution is not sandboxed; only run trusted code.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Relative path returned by create_python_script"},
            "sha256": {"type": "string", "description": "Exact SHA-256 returned by create_python_script"}}, "required": ["path", "sha256"]}}},
]


class ToolError(RuntimeError):
    """A requested tool operation could not safely complete."""


class AgentTools:
    def __init__(self, project_root=None, *, approval_input=input, output=print, ollama_url=None):
        self.root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.approval_input = approval_input
        self.output = output
        self.ollama_url = (ollama_url or os.environ.get("ACHILLUS_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.registry = {
            "list_files": self.list_files,
            "read_file": self.read_file,
            "check_ollama": self.check_ollama,
            "create_python_script": self.create_python_script,
            "run_python_script": self.run_python_script,
        }

    def _resolve(self, relative_path, *, allow_missing=False):
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ToolError("path must be non-empty relative text")
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ToolError("absolute paths and parent traversal are not allowed")
        resolved = (self.root / candidate).resolve(strict=not allow_missing)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ToolError("path resolves outside the project workspace") from exc
        relative = resolved.relative_to(self.root)
        if any(part in BLOCKED_PARTS for part in relative.parts):
            raise ToolError("access to this project path is blocked")
        if resolved.name in BLOCKED_NAMES or resolved.suffix.lower() in BLOCKED_SUFFIXES:
            raise ToolError("access to this file type is blocked")
        return resolved

    def list_files(self, arguments):
        path = arguments.get("path", ".")
        directory = self._resolve(path)
        if not directory.is_dir():
            raise ToolError("path is not a directory")
        entries = []
        for item in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            try:
                safe = self._resolve(item.relative_to(self.root).as_posix())
            except (ToolError, FileNotFoundError):
                continue
            entries.append({"name": item.name, "type": "directory" if safe.is_dir() else "file"})
            if len(entries) >= 100:
                break
        return {"path": directory.relative_to(self.root).as_posix() or ".", "entries": entries, "truncated": len(entries) == 100}

    def read_file(self, arguments):
        path = self._resolve(arguments.get("path"))
        if not path.is_file():
            raise ToolError("path is not a regular file")
        if path.stat().st_size > MAX_READ_BYTES:
            raise ToolError(f"file exceeds the {MAX_READ_BYTES}-byte read limit")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError("file is not valid UTF-8 text") from exc
        return {"path": path.relative_to(self.root).as_posix(), "content": content}

    def check_ollama(self, arguments):
        result = {}
        for endpoint, key in (("/api/tags", "models"), ("/api/ps", "running")):
            try:
                request = urllib.request.Request(self.ollama_url + endpoint, method="GET")
                with urllib.request.urlopen(request, timeout=3) as response:
                    data = json.loads(response.read(256 * 1024).decode("utf-8"))
                if key == "models":
                    result[key] = [item.get("name") for item in data.get("models", []) if isinstance(item, dict)]
                else:
                    result[key] = [item.get("name") for item in data.get("models", []) if isinstance(item, dict)]
            except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                result[key] = []
                result[key + "_error"] = str(exc)[:300]
        result["available"] = "models_error" not in result
        result["endpoint"] = self.ollama_url
        return result

    def create_python_script(self, arguments):
        filename = arguments.get("filename")
        code = arguments.get("code")
        if not isinstance(filename, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.py", filename):
            raise ToolError("filename must be a simple .py filename")
        if not isinstance(code, str) or not code.strip():
            raise ToolError("code must be non-empty text")
        if len(code.encode("utf-8")) > MAX_SCRIPT_BYTES:
            raise ToolError(f"script exceeds the {MAX_SCRIPT_BYTES}-byte limit")
        pending = self.root / ".achillus" / "scripts" / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        path = pending / filename
        if path.exists():
            raise ToolError("a script with that filename already exists; choose a new name")
        path.write_text(code, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.output("\n=== ACHILLUS PYTHON SCRIPT FOR REVIEW ===")
        self.output(f"Path: {path.relative_to(self.root).as_posix()}")
        self.output(f"SHA-256: {digest}")
        self.output(code)
        self.output("Do not execute this script unless you have reviewed and trust its contents.")
        return {"path": path.relative_to(self.root).as_posix(), "sha256": digest, "bytes": path.stat().st_size, "status": "created_not_executed"}

    def run_python_script(self, arguments):
        relative_path = arguments.get("path")
        expected_hash = arguments.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ToolError("a valid SHA-256 is required")
        path = self._resolve(relative_path)
        pending_root = (self.root / ".achillus" / "scripts" / "pending").resolve()
        try:
            path.relative_to(pending_root)
        except ValueError as exc:
            raise ToolError("only pending Achillus scripts can be executed") from exc
        if not path.is_file() or path.suffix != ".py":
            raise ToolError("script must be an existing .py file")
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ToolError("script changed since review; create/review it again")
        if path.stat().st_size > MAX_SCRIPT_BYTES:
            raise ToolError("script exceeds the execution size limit")
        self.output("\n=== ACHILLUS PYTHON EXECUTION APPROVAL ===")
        self.output(f"Script: {path.relative_to(self.root).as_posix()}")
        self.output(f"SHA-256: {actual_hash}")
        self.output("This runs with your user permissions; it is NOT sandboxed.")
        self.output("Script source:")
        self.output(path.read_text(encoding="utf-8"))
        expected = f"RUN PYTHON {actual_hash}"
        response = self.approval_input(f"\nTo execute this exact script, type:\n{expected}\n> ").strip()
        if response != expected:
            return {"status": "denied", "reason": "terminal approval not supplied"}
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            proc = subprocess.run([sys.executable, "-I", str(path)], cwd=str(self.root), env=env,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=PYTHON_TIMEOUT_SECONDS, check=False)
            return {"status": "completed" if proc.returncode == 0 else "failed", "returncode": proc.returncode,
                    "stdout": proc.stdout[-MAX_OUTPUT_CHARS:], "stderr": proc.stderr[-MAX_OUTPUT_CHARS:]}
        except subprocess.TimeoutExpired as exc:
            return {"status": "timeout", "timeout_seconds": PYTHON_TIMEOUT_SECONDS,
                    "stdout": (exc.stdout or "")[-MAX_OUTPUT_CHARS:] if isinstance(exc.stdout, str) else "",
                    "stderr": (exc.stderr or "")[-MAX_OUTPUT_CHARS:] if isinstance(exc.stderr, str) else ""}

    def dispatch(self, name, arguments):
        if name not in self.registry:
            raise ToolError(f"tool is not permitted: {name}")
        if not isinstance(arguments, dict):
            raise ToolError("tool arguments must be an object")
        return self.registry[name](arguments)
