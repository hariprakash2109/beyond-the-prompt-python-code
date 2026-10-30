import json
from datetime import date

import pytest

from ex08_tool_assistant import answer_with_tools, build_default_registry, safe_eval
from ex09_research_agent import build_notes_registry, run_research_agent
from llm_helpers import LLMReply, ScriptedLLM, ToolCall


def call(name, args, cid="c1"):
    return ToolCall(cid, name, json.dumps(args) if not isinstance(args, str) else args)


# ---------------- safe_eval ----------------
@pytest.mark.parametrize("expr,expected", [("2+3*4", 14), ("(12+8)*3", 60), ("-5+2", -3), ("2**10", 1024)])
def test_safe_eval_valid(expr, expected):
    assert safe_eval(expr) == expected


@pytest.mark.parametrize("expr", ["__import__('os').system('echo hi')", "open('x')", "2**9999",
                                  "[1,2]", "a+1", "'a'*5", "True+1", "1/0"])
def test_safe_eval_rejects_dangerous_or_invalid(expr):
    with pytest.raises((ValueError, ZeroDivisionError, SyntaxError)):
        safe_eval(expr)


# ---------------- registry ----------------
def test_registry_executes_valid_call():
    reg = build_default_registry(today=lambda: date(2026, 1, 1))
    out = reg.execute("days_until", '{"target_date": "2026-01-31"}')
    assert out == {"ok": True, "result": 30}


def test_registry_rejects_unknown_tool_bad_json_and_bad_args():
    reg = build_default_registry()
    assert not reg.execute("delete_everything", "{}")["ok"]
    assert "valid JSON" in reg.execute("calculate", "{not json")["error"]
    assert "Invalid arguments" in reg.execute("calculate", '{"expression": 5}')["error"]
    assert "Invalid arguments" in reg.execute("days_until", '{"target_date": "tomorrow"}')["error"]
    assert not reg.execute("calculate", '["a"]')["ok"]


def test_tool_failure_is_returned_not_raised():
    out = build_default_registry().execute("calculate", '{"expression": "1/0"}')
    assert out["ok"] is False and "ZeroDivisionError" in out["error"]


def test_schemas_use_flat_responses_format():
    schema = build_default_registry().schemas()[0]
    assert schema["type"] == "function" and "name" in schema and "parameters" in schema
    assert "function" not in schema


def test_answer_with_tools_round_trip():
    llm = ScriptedLLM([
        LLMReply(tool_calls=[call("calculate", {"expression": "12*12"})]),
        LLMReply(text="12 times 12 is 144."),
    ])
    assert answer_with_tools("What is 12*12?", llm, build_default_registry()) == "12 times 12 is 144."
    second_request = llm.seen_messages[1]
    outputs = [m for m in second_request if m.get("type") == "function_call_output"]
    assert json.loads(outputs[0]["output"]) == {"ok": True, "result": 144}


def test_answer_with_tools_without_tool_call():
    llm = ScriptedLLM([LLMReply(text="Hello!")])
    assert answer_with_tools("Hi", llm, build_default_registry()) == "Hello!"


# ---------------- research agent ----------------
NOTES = {
    "n1": "Python 3.12 improved error messages and f-string parsing.",
    "n2": "Virtual environments isolate project dependencies.",
}


def test_agent_happy_path_and_trace():
    llm = ScriptedLLM([
        LLMReply(tool_calls=[call("search_notes", {"query": "virtual environments"})]),
        LLMReply(tool_calls=[call("read_note", {"note_id": "n2"}, "c2")]),
        LLMReply(text="They isolate dependencies [n2]."),
    ])
    run = run_research_agent("What do venvs do?", llm, build_notes_registry(NOTES))
    assert run.stop_reason == "answered" and run.iterations == 3
    assert run.notes_read == {"n2"}
    assert [t["tool"] for t in run.trace] == ["search_notes", "read_note"]


def test_agent_stops_at_iteration_limit():
    forever = [LLMReply(tool_calls=[call("search_notes", {"query": "python"}, f"c{i}")]) for i in range(10)]
    run = run_research_agent("loop", ScriptedLLM(forever), build_notes_registry(NOTES), max_iterations=3)
    assert run.stop_reason == "max_iterations" and run.iterations == 3 and run.final_answer is None


def test_agent_stops_at_tool_budget():
    many = [LLMReply(tool_calls=[call("search_notes", {"query": "python"}, f"a{i}"),
                                 call("search_notes", {"query": "python"}, f"b{i}")]) for i in range(5)]
    run = run_research_agent("x", ScriptedLLM(many), build_notes_registry(NOTES),
                             max_iterations=5, max_tool_calls=3)
    assert run.stop_reason == "tool_budget"


def test_agent_stops_on_timeout_using_fake_clock():
    ticks = iter([0, 0, 999, 999, 999])
    run = run_research_agent("x", ScriptedLLM([LLMReply(tool_calls=[call("search_notes", {"query": "py"})]),
                                               LLMReply(text="never reached")]),
                             build_notes_registry(NOTES), timeout_seconds=10, clock=lambda: next(ticks))
    assert run.stop_reason == "timeout"


def test_agent_recovers_from_malformed_and_unknown_tool_calls():
    llm = ScriptedLLM([
        LLMReply(tool_calls=[call("search_notes", "{broken json")]),
        LLMReply(tool_calls=[call("rm_rf", {})]),
        LLMReply(tool_calls=[call("read_note", {"note_id": "../../etc/passwd"})]),
        LLMReply(text="I could not complete the search."),
    ])
    run = run_research_agent("x", llm, build_notes_registry(NOTES))
    assert run.stop_reason == "answered"
    assert [t["ok"] for t in run.trace] == [False, False, False]


def test_agent_feeds_errors_back_to_the_model():
    llm = ScriptedLLM([LLMReply(tool_calls=[call("read_note", {"note_id": "missing"})]),
                       LLMReply(text="done")])
    run_research_agent("x", llm, build_notes_registry(NOTES))
    last_request = llm.seen_messages[1]
    output = [m for m in last_request if m.get("type") == "function_call_output"][0]
    assert json.loads(output["output"])["ok"] is False
