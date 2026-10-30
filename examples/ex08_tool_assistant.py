"""Project 5: a tool-using assistant (Chapter 8).

Key idea: the model only SUGGESTS a tool call. Our code decides whether to
run it. Every tool is registered in an allowlist, has a JSON schema for the
model, and a Pydantic model that validates the arguments before execution.
"""
from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

try:
    from llm_helpers import LLM, LLMReply
except ImportError:  # pragma: no cover
    from examples.llm_helpers import LLM, LLMReply

# ---------- safe calculator (never use eval on model output) ----------
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_eval(expression: str) -> float:
    """Evaluate +, -, *, /, %, ** on plain numbers. Anything else is rejected."""
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.operand))
        raise ValueError("Unsupported expression")
    if len(expression) > 200:
        raise ValueError("Expression too long")
    return walk(ast.parse(expression, mode="eval"))


# ---------- argument models ----------
class CalculateArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200,
                            description="Arithmetic such as '(12 + 8) * 3'")


class WordCountArgs(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class DaysUntilArgs(BaseModel):
    target_date: date = Field(description="ISO date, e.g. 2027-01-31")


# ---------- registry ----------
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[Any], Any]
    needs_approval: bool = False

    def schema(self) -> dict[str, Any]:
        """Flat function-tool definition used by the OpenAI Responses API."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }


class ToolRegistry:
    """The allowlist. If a tool is not registered here, it cannot run."""

    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments_json: str) -> dict[str, Any]:
        """Validate and run a tool. Never raises: returns ok/error dicts."""
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"Unknown tool '{name}'"}
        try:
            raw_args = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": "Arguments were not valid JSON"}
        if not isinstance(raw_args, dict):
            return {"ok": False, "error": "Arguments must be a JSON object"}
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return {"ok": False, "error": f"Invalid arguments: {exc.errors(include_url=False)}"}
        try:
            return {"ok": True, "result": tool.func(args)}
        except Exception as exc:  # a failing tool must not crash the assistant
            return {"ok": False, "error": f"Tool failed: {type(exc).__name__}: {exc}"}


def build_default_registry(today: Callable[[], date] = date.today) -> ToolRegistry:
    return ToolRegistry([
        Tool("calculate", "Evaluate an arithmetic expression exactly.",
             CalculateArgs, lambda a: safe_eval(a.expression)),
        Tool("word_count", "Count the words in a piece of text.",
             WordCountArgs, lambda a: len(a.text.split())),
        Tool("days_until", "Number of days from today until a given date.",
             DaysUntilArgs, lambda a: (a.target_date - today()).days),
    ])


def answer_with_tools(question: str, llm: LLM, registry: ToolRegistry) -> str:
    """One round trip: ask, run any requested tools, ask again for the final text."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    first = llm.generate(messages, tools=registry.schemas())
    if not first.tool_calls:
        return first.text

    for call in first.tool_calls:
        messages.append({"type": "function_call", "call_id": call.call_id,
                         "name": call.name, "arguments": call.arguments_json})
        outcome = registry.execute(call.name, call.arguments_json)
        messages.append({"type": "function_call_output", "call_id": call.call_id,
                         "output": json.dumps(outcome, default=str)})
    # No tools offered on the second call: this example allows only one round.
    return llm.generate(messages).text
