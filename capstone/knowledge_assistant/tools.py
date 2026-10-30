"""Narrow, validated tools. The model may REQUEST them; this module decides.

Tools
-----
search_knowledge_base  read-only   semantic search limited to what the user may read
get_document_info      read-only   metadata for one document the user may read
save_note              SENSITIVE   writes a file, so it needs human approval

Nothing here executes model-generated code or shell commands.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from .access import User, can_read
from .rag import RagService
from .vector_store import VectorStore


class SearchArgs(BaseModel):
    query: str = Field(min_length=3, max_length=300, description="What to look for")
    top_k: int = Field(default=3, ge=1, le=5)


class DocInfoArgs(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,60}$", description="Document id, e.g. internship_policy")


class SaveNoteArgs(BaseModel):
    title: str = Field(min_length=3, max_length=80)
    body: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[BaseModel, User], Any]
    needs_approval: bool = False

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "name": self.name, "description": self.description,
                "parameters": self.args_model.model_json_schema()}


@dataclass
class Validated:
    ok: bool
    args: BaseModel | None = None
    error: str = ""


class ToolRegistry:
    """Allowlist of tools. Unknown names are rejected, never guessed."""

    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def validate(self, name: str, arguments_json: str) -> Validated:
        tool = self._tools.get(name)
        if tool is None:
            return Validated(False, error=f"Unknown tool '{name}'")
        try:
            raw = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return Validated(False, error="Arguments were not valid JSON")
        if not isinstance(raw, dict):
            return Validated(False, error="Arguments must be a JSON object")
        try:
            return Validated(True, args=tool.args_model.model_validate(raw))
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return Validated(False, error=f"Invalid arguments: {problems}")

    def run(self, name: str, args: BaseModel, user: User) -> dict[str, Any]:
        """Execute an ALREADY VALIDATED call. Failures are returned, not raised."""
        tool = self._tools[name]
        try:
            return {"ok": True, "result": tool.func(args, user)}
        except Exception as exc:
            return {"ok": False, "error": f"Tool failed: {type(exc).__name__}: {exc}"}


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "note"


def build_registry(rag: RagService, store: VectorStore, notes_dir: Path) -> ToolRegistry:
    notes_dir = Path(notes_dir)

    def search_knowledge_base(args: SearchArgs, user: User) -> list[dict[str, Any]]:
        hits = rag.retrieve(args.query, user, k=args.top_k)
        return [{"chunk_id": h.chunk.chunk_id, "title": h.chunk.title, "source": h.chunk.source,
                 "score": round(h.score, 3), "snippet": h.chunk.text[:240]} for h in hits]

    def get_document_info(args: DocInfoArgs, user: User) -> dict[str, Any]:
        chunks = store.get_document_chunks(args.doc_id)
        # Same message for "missing" and "forbidden": do not reveal that restricted documents exist.
        if not chunks or not can_read(user.role, chunks[0].access):
            raise LookupError(f"No document '{args.doc_id}' is available to you")
        first = chunks[0]
        return {"doc_id": first.doc_id, "title": first.title, "source": first.source,
                "access": first.access, "chunks": len(chunks)}

    def save_note(args: SaveNoteArgs, user: User) -> dict[str, str]:
        notes_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{_slug(args.title)}-{uuid.uuid4().hex[:8]}.md"   # never overwrites, never user-controlled path
        (notes_dir / filename).write_text(
            f"# {args.title}\n\nSaved for: {user.name}\n\n{args.body}\n", encoding="utf-8")
        return {"saved_as": filename}

    return ToolRegistry([
        Tool("search_knowledge_base", "Search the knowledge base for passages relevant to a query.",
             SearchArgs, search_knowledge_base),
        Tool("get_document_info", "Get title, access level and size of one document by its id.",
             DocInfoArgs, get_document_info),
        Tool("save_note", "Save a short note to the user's notes folder. Requires human approval.",
             SaveNoteArgs, save_note, needs_approval=True),
    ])
