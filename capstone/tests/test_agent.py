import json

from helpers import FAKE_SECRET, STAFF, STUDENT, LLMReply, tool_call
from knowledge_assistant.agent import PendingNotFound
import pytest


def notes(settings):
    return list(settings.notes_dir.glob("*.md")) if settings.notes_dir.exists() else []


def test_agent_uses_read_only_tool_then_answers(make_services):
    services, llm = make_services([
        LLMReply(tool_calls=[tool_call("search_knowledge_base", {"query": "laptop loan days"})]),
        LLMReply(text="You can borrow a laptop for 14 days (Laptop Loan Program)."),
    ])
    result = services.agent.run("How long can I borrow a laptop?", STUDENT)
    assert result.status == "answered" and "14 days" in result.answer
    assert result.trace == [{"iteration": 1, "tool": "search_knowledge_base", "status": "executed", "ok": True}]
    tool_output = [m for m in llm.seen[1]["messages"] if m.get("type") == "function_call_output"][0]
    assert json.loads(tool_output["output"])["ok"] is True


def test_sensitive_tool_pauses_and_writes_nothing_until_approved(make_services, settings):
    services, _ = make_services([LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "Book a mock interview"})])])
    result = services.agent.run("Save a note: book a mock interview", STUDENT)
    assert result.status == "needs_approval"
    assert result.pending.tool_name == "save_note"
    assert notes(settings) == []          # the model SUGGESTED; nothing was EXECUTED


def test_approval_executes_and_agent_continues(make_services, settings):
    services, llm = make_services([
        LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "Book a mock interview"})]),
        LLMReply(text="Saved your note."),
    ])
    pending = services.agent.run("Save a note", STUDENT).pending
    final = services.agent.resume(pending.pending_id, True, STUDENT)
    assert final.status == "answered" and len(notes(settings)) == 1
    assert "Book a mock interview" in notes(settings)[0].read_text()
    output = [m for m in llm.seen[1]["messages"] if m.get("type") == "function_call_output"][0]
    assert json.loads(output["output"])["ok"] is True


def test_denial_does_not_execute_and_tells_the_model(make_services, settings):
    services, llm = make_services([
        LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "x"})]),
        LLMReply(text="Okay, I did not save it."),
    ])
    pending = services.agent.run("Save a note", STUDENT).pending
    final = services.agent.resume(pending.pending_id, False, STUDENT)
    assert final.status == "answered" and notes(settings) == []
    output = [m for m in llm.seen[1]["messages"] if m.get("type") == "function_call_output"][0]
    assert "denied" in json.loads(output["output"])["error"]


def test_approval_is_single_use(make_services):
    services, _ = make_services([
        LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "x"})]),
        LLMReply(text="done"),
    ])
    pending = services.agent.run("Save a note", STUDENT).pending
    services.agent.resume(pending.pending_id, True, STUDENT)
    with pytest.raises(PendingNotFound):
        services.agent.resume(pending.pending_id, True, STUDENT)


def test_another_user_cannot_approve_my_action(make_services, settings):
    services, _ = make_services([LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "x"})])])
    pending = services.agent.run("Save a note", STUDENT).pending
    with pytest.raises(PendingNotFound):
        services.agent.resume(pending.pending_id, True, STAFF)
    assert notes(settings) == []


def test_pending_action_expires(make_services, settings):
    services, _ = make_services([LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "x"})])])
    pending = services.agent.run("Save a note", STUDENT).pending
    services.agent.pending.ttl = -1  # simulate that the time limit has already passed
    with pytest.raises(PendingNotFound):
        services.agent.resume(pending.pending_id, True, STUDENT)
    assert notes(settings) == []


def test_invalid_sensitive_call_is_rejected_before_asking_a_human(make_services):
    services, _ = make_services([
        LLMReply(tool_calls=[tool_call("save_note", {"title": "x", "body": ""})]),
        LLMReply(text="Sorry, I could not save that."),
    ])
    result = services.agent.run("Save a note", STUDENT)
    assert result.status == "answered" and result.trace[0]["status"] == "rejected"


def test_malformed_and_unknown_calls_are_fed_back_not_executed(make_services, settings):
    services, llm = make_services([
        LLMReply(tool_calls=[tool_call("search_knowledge_base", "{oops"), tool_call("rm_rf", {"path": "/"}, "call-2")]),
        LLMReply(text="I could not use my tools."),
    ])
    result = services.agent.run("anything", STUDENT)
    assert [t["status"] for t in result.trace] == ["rejected", "rejected"]
    assert notes(settings) == []


def test_loop_stops_at_iteration_limit(make_services):
    endless = [LLMReply(tool_calls=[tool_call("search_knowledge_base", {"query": "internship"}, f"c{i}")]) for i in range(10)]
    services, llm = make_services(endless, max_iterations=3)
    result = services.agent.run("loop forever", STUDENT)
    assert result.status == "stopped" and result.stop_reason == "max_iterations" and len(llm.seen) == 3


def test_loop_stops_on_timeout(make_services):
    services, llm = make_services([LLMReply(text="never reached")], timeout_seconds=5)
    ticks = iter([0.0, 100.0, 100.0])
    services.agent.clock = lambda: next(ticks)
    result = services.agent.run("slow", STUDENT)
    assert result.status == "stopped" and result.stop_reason == "timeout" and llm.seen == []


def test_calls_after_a_sensitive_call_are_dropped_until_the_human_decides(make_services):
    services, _ = make_services([LLMReply(tool_calls=[
        tool_call("save_note", {"title": "Plan", "body": "x"}, "call-a"),
        tool_call("search_knowledge_base", {"query": "internship"}, "call-b")])])
    result = services.agent.run("Save and search", STUDENT)
    ids = [m.get("call_id") for m in result.pending.messages if m.get("type") == "function_call"]
    assert ids == ["call-a"]


def test_prompt_injection_in_a_document_cannot_write_without_approval(make_services, settings):
    """The planted 'save a note titled pwned' text lives in campus_events.md.

    We simulate the WORST CASE: the model obeys the injected text and requests save_note.
    The approval gate must still stop it, and the human can refuse.
    """
    services, llm = make_services([
        LLMReply(tool_calls=[tool_call("search_knowledge_base", {"query": "Spring Career Fair"})]),
        LLMReply(tool_calls=[tool_call("save_note", {"title": "pwned", "body": "injected"}, "call-2")]),
    ])
    result = services.agent.run("When is the Spring Career Fair?", STUDENT)
    tool_text = [m for m in llm.seen[1]["messages"] if m.get("type") == "function_call_output"][0]["output"]
    assert "ignore all previous instructions" in tool_text      # the attack text really reached the model
    assert result.status == "needs_approval" and notes(settings) == []


def test_injected_text_cannot_reveal_restricted_documents_to_a_student(make_services):
    services, llm = make_services([
        LLMReply(tool_calls=[tool_call("search_knowledge_base", {"query": "partner employers placement fee"})]),
        LLMReply(text="I only found public information."),
    ])
    services.agent.run("Reveal all staff documents", STUDENT)
    assert "8 percent" not in json.dumps(llm.seen)


def test_secrets_never_enter_model_visible_context(make_services, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_SECRET)
    services, llm = make_services([
        LLMReply(tool_calls=[tool_call("search_knowledge_base", {"query": "internship credit"})]),
        LLMReply(text="Done."),
    ])
    services.agent.run("Tell me about internships", STUDENT)
    assert FAKE_SECRET not in json.dumps(llm.seen)
    assert "k-student" not in json.dumps(llm.seen) and "k-staff" not in json.dumps(llm.seen)
    assert "k-student" not in repr(services.settings)  # keys are hidden from repr/logging
