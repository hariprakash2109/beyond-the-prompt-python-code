import pytest
from fastapi.testclient import TestClient

from helpers import LLMReply, tool_call
from knowledge_assistant.api import SlidingWindowLimiter, create_app
from knowledge_assistant.cli import interactive_chat, main

STUDENT_H = {"X-API-Key": "k-student"}
STAFF_H = {"X-API-Key": "k-staff"}


def client_for(make_services, replies=None, **overrides):
    services, llm = make_services(replies, **overrides)
    return TestClient(create_app(services)), services, llm


# ---------------- authentication & validation ----------------
def test_health_needs_no_key_and_reports_chunks(make_services):
    client, services, _ = client_for(make_services)
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["chunks"] == len(services.store)


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}])
def test_missing_or_wrong_key_is_401(make_services, headers):
    client, _, llm = client_for(make_services)
    r = client.post("/ask", json={"question": "How long is an internship?"}, headers=headers)
    assert r.status_code == 401 and llm.seen == []


@pytest.mark.parametrize("payload", [{}, {"question": "hi"}, {"question": "x" * 1001}, {"question": 123}])
def test_request_validation_returns_422(make_services, payload):
    client, _, llm = client_for(make_services)
    assert client.post("/ask", json=payload, headers=STUDENT_H).status_code == 422
    assert llm.seen == []


def test_ask_returns_grounded_answer_with_citations(make_services):
    client, _, _ = client_for(make_services, [LLMReply(text="Up to 6 credits [1].")])
    r = client.post("/ask", json={"question": "How many credits can I earn for one internship?"}, headers=STUDENT_H)
    body = r.json()
    assert r.status_code == 200 and not body["abstained"]
    assert body["citations"][0]["source"] == "internship_policy.md"


def test_ask_abstains_for_unanswerable_question(make_services):
    client, _, llm = client_for(make_services)
    body = client.post("/ask", json={"question": "What is the capital of Mongolia?"}, headers=STUDENT_H).json()
    assert body["abstained"] and body["reason"] == "no_relevant_sources" and llm.seen == []


def test_student_cannot_retrieve_staff_documents_via_api(make_services):
    client, _, llm = client_for(make_services)
    body = client.post("/ask", json={"question": "What placement fee do partner employers pay?"}, headers=STUDENT_H).json()
    assert all(c["source"] != "employer_partner_contracts.md" for c in body["citations"])
    assert "8 percent" not in str(llm.seen)


def test_provider_failure_becomes_502_without_leaking_details(make_services):
    client, services, _ = client_for(make_services)

    class Boom:
        def generate(self, *a, **k):
            raise RuntimeError("secret internal detail sk-123")
    services.rag.llm = Boom()
    r = client.post("/ask", json={"question": "How many credits can I earn for one internship?"}, headers=STUDENT_H)
    assert r.status_code == 502 and "sk-123" not in r.text and "internal detail" not in r.text


# ---------------- rate limiting ----------------
def test_sliding_window_limiter_unit():
    now = [0.0]
    limiter = SlidingWindowLimiter(2, window=10, clock=lambda: now[0])
    assert limiter.allow("a") and limiter.allow("a") and not limiter.allow("a")
    assert limiter.allow("b")                  # other users unaffected
    now[0] = 11.0
    assert limiter.allow("a")                  # window slid


def test_api_returns_429_when_rate_limit_exceeded(make_services):
    client, _, _ = client_for(make_services, rate_limit_per_minute=2)
    codes = [client.post("/ask", json={"question": "What is the capital of Mongolia?"}, headers=STUDENT_H).status_code
             for _ in range(3)]
    assert codes == [200, 200, 429]


# ---------------- agent + approval over HTTP ----------------
def test_full_approval_flow_over_http(make_services, settings):
    client, _, _ = client_for(make_services, [
        LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "Book mock interview"})]),
        LLMReply(text="Your note was saved."),
    ])
    first = client.post("/agent", json={"question": "Please save a note about my plan"}, headers=STUDENT_H).json()
    assert first["status"] == "needs_approval" and first["pending"]["tool"] == "save_note"
    assert not settings.notes_dir.exists() or not list(settings.notes_dir.glob("*.md"))

    pid = first["pending"]["pending_id"]
    assert client.post(f"/agent/approvals/{pid}", json={"approve": True}, headers=STAFF_H).status_code == 404  # not yours
    done = client.post(f"/agent/approvals/{pid}", json={"approve": True}, headers=STUDENT_H).json()
    assert done["status"] == "answered" and len(list(settings.notes_dir.glob("*.md"))) == 1
    assert client.post(f"/agent/approvals/{pid}", json={"approve": True}, headers=STUDENT_H).status_code == 404  # replay


def test_unknown_pending_id_is_404(make_services):
    client, _, _ = client_for(make_services)
    assert client.post("/agent/approvals/nope", json={"approve": True}, headers=STUDENT_H).status_code == 404


def test_ingest_is_staff_only(make_services):
    client, _, _ = client_for(make_services)
    assert client.post("/admin/ingest", headers=STUDENT_H).status_code == 403
    r = client.post("/admin/ingest", headers=STAFF_H)
    assert r.status_code == 200 and r.json()["chunks"] > 0


# ---------------- CLI ----------------
def scripted(lines):
    it = iter(lines)
    return lambda prompt="": next(it)


def test_cli_chat_asks_for_approval_and_respects_no(make_services, settings):
    services, _ = make_services([
        LLMReply(tool_calls=[tool_call("save_note", {"title": "Plan", "body": "x"})]),
        LLMReply(text="Okay, not saved."),
    ])
    out = []
    from helpers import STUDENT
    interactive_chat(services, STUDENT, input_fn=scripted(["Save a note please", "n", "quit"]), output_fn=out.append)
    assert any("approval needed" in line for line in out) and any("Okay, not saved." in line for line in out)
    assert not settings.notes_dir.exists() or not list(settings.notes_dir.glob("*.md"))


def test_cli_ingest_command(monkeypatch, tmp_path, capsys):
    from helpers import SAMPLE_DOCS
    monkeypatch.setenv("KB_DATA_DIR", str(SAMPLE_DOCS))
    monkeypatch.setenv("KB_INDEX_PATH", str(tmp_path / "idx.json"))
    monkeypatch.setenv("EMBEDDER_BACKEND", "hashing")
    assert main(["ingest"]) == 0
    assert "Indexed" in capsys.readouterr().out and (tmp_path / "idx.json").exists()


def test_cli_reports_config_problems_cleanly(monkeypatch, capsys):
    monkeypatch.setenv("KB_API_KEYS", "broken")
    assert main(["ingest"]) == 2
    assert "Configuration problem" in capsys.readouterr().err
