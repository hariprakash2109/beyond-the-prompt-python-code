"""Shared constants and small helpers for the capstone tests."""
import json
import sys
from pathlib import Path

CAPSTONE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAPSTONE))

from knowledge_assistant.access import User  # noqa: E402
from knowledge_assistant.llm import LLMReply, ToolCall  # noqa: E402,F401

SAMPLE_DOCS = CAPSTONE / "sample_data" / "docs"
EVAL_FILE = CAPSTONE / "sample_data" / "eval_questions.json"

STUDENT = User("alice", "student")
STAFF = User("bob", "staff")
FAKE_SECRET = "test-secret-not-a-real-api-key"


def tool_call(name, args, call_id="call-1"):
    return ToolCall(call_id, name, args if isinstance(args, str) else json.dumps(args))
