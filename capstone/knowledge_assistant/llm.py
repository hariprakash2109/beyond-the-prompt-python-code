"""Model access behind one method: generate(messages, tools) -> LLMReply.

OpenAIResponsesLLM follows OpenAI's official Responses API function-calling
format (flat tool definitions; function_call / function_call_output items).
Its live network path was NOT executed while writing this book. Everything
else in the project is tested with ScriptedLLM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import ConfigError, Settings


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments_json: str


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def generate(self, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None) -> LLMReply: ...


class OpenAIResponsesLLM:
    def __init__(self, model: str, client: Any | None = None, timeout: float = 30.0):
        if not model:
            raise ConfigError("LLM_MODEL is not set. Choose a model available to your account.")
        if client is None:
            from openai import OpenAI
            client = OpenAI(timeout=timeout)  # reads OPENAI_API_KEY itself; the key never enters prompts
        self.model, self.client = model, client

    def generate(self, messages, tools=None) -> LLMReply:
        kwargs: dict[str, Any] = {"model": self.model, "input": messages}
        if tools:
            kwargs["tools"] = tools
        response = self.client.responses.create(**kwargs)
        calls = [ToolCall(i.call_id, i.name, i.arguments)
                 for i in response.output if getattr(i, "type", None) == "function_call"]
        return LLMReply(text=response.output_text or "", tool_calls=calls)


class ScriptedLLM:
    """Replays prepared replies. `seen` records every request for assertions."""

    def __init__(self, replies: list[LLMReply]):
        self._replies = list(replies)
        self.seen: list[dict[str, Any]] = []

    def generate(self, messages, tools=None) -> LLMReply:
        self.seen.append({"messages": [dict(m) for m in messages], "tools": tools})
        if not self._replies:
            raise AssertionError("ScriptedLLM has no more replies")
        return self._replies.pop(0)


def build_llm(settings: Settings) -> LLM:
    if not os.environ.get("OPENAI_API_KEY"):
        raise ConfigError("OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return OpenAIResponsesLLM(settings.llm_model)
