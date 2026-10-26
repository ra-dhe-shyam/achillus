"""Bounded tool orchestration using provider-native tool calls when available."""
import json
import logging

from agent_tools import AgentTools, TOOL_SPECS
from model_interface import _messages

logger = logging.getLogger("personal_ai.cognition")
MAX_STEPS = 3


class CognitiveLoop:
    """Let the provider request allowlisted tools; execute and return observations."""

    def __init__(self, model, max_steps=MAX_STEPS, tool_registry=None):
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 8:
            raise ValueError("max_steps must be an integer between 1 and 8")
        self.model = model
        self.max_steps = max_steps
        self.agent_tools = AgentTools()
        self.tools = tool_registry or self.agent_tools.registry

    @staticmethod
    def _tool_schemas(registry):
        permitted = set(registry)
        return [item for item in TOOL_SPECS if item["function"]["name"] in permitted]

    def run(self, user_message, memories):
        """Use one model call for a direct answer; continue only after tool calls."""
        messages = _messages(user_message, memories)
        schemas = self._tool_schemas(self.tools)
        observations = []
        active_model = self.model

        for _ in range(self.max_steps):
            result = active_model.generate_with_tools(messages, schemas)
            if isinstance(result, dict) and result.get("_provider_model") is not None:
                active_model = result["_provider_model"]
            if not isinstance(result, dict) or result.get("type") != "tool_call":
                if isinstance(result, dict) and isinstance(result.get("content"), str) and result["content"].strip():
                    return result["content"].strip()
                raise RuntimeError("Model returned an invalid tool-aware response")

            name = result.get("name")
            arguments = result.get("arguments", {})
            assistant_message = result.get("assistant_message")
            if not isinstance(assistant_message, dict):
                raise RuntimeError("Tool call omitted the provider assistant message")
            if name not in self.tools:
                logger.warning("Rejected unregistered tool request: %r", name)
                return "I couldn't perform that action because the requested tool is not permitted."
            try:
                tool_result = self.tools[name](arguments) if self.tools is not self.agent_tools.registry else self.agent_tools.dispatch(name, arguments)
            except Exception as exc:
                # Tool failures become observations; they are never represented as success.
                logger.exception("Tool execution failed: %s", name)
                tool_result = {"error": str(exc)[:500], "status": "failed"}
            observations.append({"tool": name, "result": tool_result})
            messages.append(assistant_message)
            messages.append(active_model.tool_result_message(result, tool_result))

        # Budget is exhausted: no more tools. Ask the model for a final response
        # without tool definitions, so a further action cannot be executed.
        messages.append({"role": "user", "content": (
            "The tool-call budget is exhausted. Give the best truthful answer now "
            "using the observations below. Do not claim unperformed actions.\n"
            + json.dumps(observations, ensure_ascii=False)
        )})
        final = active_model.generate_from_messages(messages)
        if not isinstance(final, str) or not final.strip():
            raise RuntimeError("Model returned an empty final answer after tool use")
        return final.strip()
