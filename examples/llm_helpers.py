"""Small, provider-neutral helpers shared by the book's examples.

The rest of the code talks to an object with one method:

    llm.generate(messages, tools=None) -> LLMReply

Two implementations are provided:

* OpenAIResponsesLLM - calls the official OpenAI Python SDK (Responses API).
* ScriptedLLM        - returns pre-written replies. Used in automated tests so
                       that no network access or API key is needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A *request* from the model to run a tool. Nothing has run yet."""
    call_id: str
    name: str
    arguments_json: str  # raw JSON text exactly as the model produced it


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def generate(self, messages: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None) -> LLMReply: ...


class OpenAIResponsesLLM:
    """Thin adapter over client.responses.create (official OpenAI SDK).

    NOTE: the live network path of this class has not been executed by the
    author of this manuscript (no API key was available). Its request and
    response shapes follow OpenAI's official function-calling guide. The
    automated tests in this repository use ScriptedLLM instead.
    """

    def __init__(self, model: str | None = None, client: Any | None = None):
        self.model = model or os.environ.get("LLM_MODEL", "")
        if not self.model:
            raise RuntimeError(
                "No model configured. Set LLM_MODEL in your .env file to a "
                "model that is available to your account."
            )
        if client is None:
            from openai import OpenAI  # imported lazily so tests need no key
            client = OpenAI()  # reads OPENAI_API_KEY from the environment
        self.client = client

    def generate(self, messages, tools=None) -> LLMReply:
        kwargs: dict[str, Any] = {"model": self.model, "input": messages}
        if tools:
            kwargs["tools"] = tools
        response = self.client.responses.create(**kwargs)

        calls: list[ToolCall] = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                calls.append(ToolCall(item.call_id, item.name, item.arguments))
        return LLMReply(text=response.output_text or "", tool_calls=calls)


class ScriptedLLM:
    """A fake model that replays a fixed list of replies, in order."""

    def __init__(self, replies: list[LLMReply]):
        self._replies = list(replies)
        self.seen_messages: list[list[dict[str, Any]]] = []

    def generate(self, messages, tools=None) -> LLMReply:
        self.seen_messages.append(list(messages))
        if not self._replies:
            raise AssertionError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)
