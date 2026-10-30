"""FastAPI interface: authentication, validation, rate limiting and logging."""
from __future__ import annotations

import hmac
import logging
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .access import User
from .agent import AgentResult, PendingNotFound
from .config import Settings
from .rag import RagResult
from .services import Services, build_services

log = logging.getLogger("knowledge_assistant")


# ---------- request / response models ----------
class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class ApprovalRequest(BaseModel):
    approve: bool


class CitationOut(BaseModel):
    chunk_id: str
    title: str
    source: str
    score: float


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    reason: str
    citations: list[CitationOut]


class PendingOut(BaseModel):
    pending_id: str
    tool: str
    arguments: dict


class AgentResponse(BaseModel):
    status: str
    answer: str | None = None
    stop_reason: str | None = None
    pending: PendingOut | None = None
    trace: list[dict]


class IngestResponse(BaseModel):
    chunks: int


class SlidingWindowLimiter:
    """At most `limit` requests per user per `window` seconds (in memory, single process)."""

    def __init__(self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, who: str) -> bool:
        now, q = self.clock(), self._events[who]
        while q and now - q[0] > self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True


def _agent_response(result: AgentResult) -> AgentResponse:
    pending = None
    if result.pending:
        pending = PendingOut(pending_id=result.pending.pending_id, tool=result.pending.tool_name,
                             arguments=result.pending.arguments)
    return AgentResponse(status=result.status, answer=result.answer, stop_reason=result.stop_reason,
                         pending=pending, trace=result.trace)


def _ask_response(result: RagResult) -> AskResponse:
    return AskResponse(
        answer=result.answer, abstained=result.abstained, reason=result.reason,
        citations=[CitationOut(chunk_id=c.chunk_id, title=c.title, source=c.source, score=round(c.score, 3))
                   for c in result.citations])


def create_app(services: Services, clock: Callable[[], float] = time.monotonic) -> FastAPI:
    app = FastAPI(title="Knowledge Assistant", version="1.0.0")
    app.state.services = services
    limiter = SlidingWindowLimiter(services.settings.rate_limit_per_minute, clock=clock)

    def authenticate(x_api_key: str | None = Header(default=None)) -> User:
        if not x_api_key:
            raise HTTPException(401, "Missing X-API-Key header")
        match: User | None = None
        for key, user in app.state.services.settings.api_keys.items():
            if hmac.compare_digest(key.encode(), x_api_key.encode()):  # constant-time comparison
                match = user
        if match is None:
            raise HTTPException(401, "Invalid API key")
        if not limiter.allow(match.name):
            raise HTTPException(429, "Too many requests. Please slow down.")
        return match

    def provider_guard(call: Callable[[], object]):
        try:
            return call()
        except (HTTPException, PendingNotFound):
            raise  # expected, already-handled conditions pass through unchanged
        except Exception:
            log.exception("model provider or internal error")
            raise HTTPException(502, "The AI service failed. Please try again later.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "chunks": len(app.state.services.store)}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: QuestionRequest, user: User = Depends(authenticate)) -> AskResponse:
        started = time.monotonic()
        result = provider_guard(lambda: app.state.services.rag.ask(body.question, user))
        log.info("ask user=%s abstained=%s chars=%d ms=%d", user.name, result.abstained,
                 len(body.question), (time.monotonic() - started) * 1000)
        return _ask_response(result)

    @app.post("/agent", response_model=AgentResponse)
    def agent(body: QuestionRequest, user: User = Depends(authenticate)) -> AgentResponse:
        result = provider_guard(lambda: app.state.services.agent.run(body.question, user))
        log.info("agent user=%s status=%s steps=%d", user.name, result.status, len(result.trace))
        return _agent_response(result)

    @app.post("/agent/approvals/{pending_id}", response_model=AgentResponse)
    def approve(pending_id: str, body: ApprovalRequest, user: User = Depends(authenticate)) -> AgentResponse:
        try:
            result = provider_guard(lambda: app.state.services.agent.resume(pending_id, body.approve, user))
        except PendingNotFound:
            raise HTTPException(404, "No such pending action")
        log.info("approval user=%s approved=%s status=%s", user.name, body.approve, result.status)
        return _agent_response(result)

    @app.post("/admin/ingest", response_model=IngestResponse)
    def reingest(user: User = Depends(authenticate)) -> IngestResponse:
        if user.role != "staff":
            raise HTTPException(403, "Only staff may rebuild the index")
        old = app.state.services
        app.state.services = provider_guard(lambda: build_services(
            old.settings, llm=old.agent.llm, embedder=old.embedder, rebuild=True))
        return IngestResponse(chunks=len(app.state.services.store))

    return app


def create_app_from_env() -> FastAPI:
    """Entry point for uvicorn: `uvicorn knowledge_assistant.api:create_app_from_env --factory`."""
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app(build_services(Settings.from_env()))
