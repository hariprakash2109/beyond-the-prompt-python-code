"""Project 1: a command-line AI assistant (Chapter 3).

Run:
    python examples/ex03_cli_assistant.py

Requires OPENAI_API_KEY and LLM_MODEL in your .env file.
Type 'quit' to exit, 'reset' to clear the conversation.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv

MAX_HISTORY_MESSAGES = 20  # keep the request (and the bill) bounded

SYSTEM_PROMPT = (
    "You are a patient programming tutor for beginners. "
    "Answer in plain English, keep answers under 200 words, and say "
    "'I am not sure' when you are not sure."
)


def build_client() -> Any:
    """Create the SDK client, failing early with a friendly message."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing. Copy .env.example to .env first.")
    from openai import OpenAI
    return OpenAI()


def trim_history(history: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep only the most recent messages."""
    return history[-MAX_HISTORY_MESSAGES:]


def ask_model(client: Any, model: str, history: list[dict[str, str]]) -> tuple[str, int | None]:
    """Send the conversation to the model. Returns (reply_text, total_tokens)."""
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=trim_history(history),
    )
    usage = getattr(response, "usage", None)
    total = getattr(usage, "total_tokens", None) if usage else None
    return response.output_text, total


def chat_loop(client: Any, model: str, input_fn=input, output_fn=print) -> list[dict[str, str]]:
    """The read-ask-print loop. input_fn/output_fn are injectable for tests."""
    history: list[dict[str, str]] = []
    session_tokens = 0
    output_fn("AI tutor ready. Type 'quit' to exit, 'reset' to start over.")
    while True:
        try:
            user_text = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() == "quit":
            break
        if user_text.lower() == "reset":
            history.clear()
            output_fn("(conversation cleared)")
            continue
        history.append({"role": "user", "content": user_text})
        try:
            reply, tokens = ask_model(client, model, history)
        except Exception as exc:  # network, auth, rate limit, bad model name...
            history.pop()  # do not keep a question that never got an answer
            output_fn(f"[error] {type(exc).__name__}: {exc}")
            continue
        history.append({"role": "assistant", "content": reply})
        if tokens:
            session_tokens += tokens
        output_fn(f"ai> {reply}")
        if tokens:
            output_fn(f"    (tokens this turn: {tokens}, session total: {session_tokens})")
    return history


def main() -> int:
    load_dotenv()
    model = os.environ.get("LLM_MODEL", "")
    if not model:
        raise SystemExit("LLM_MODEL is missing. Set it in your .env file.")
    chat_loop(build_client(), model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
