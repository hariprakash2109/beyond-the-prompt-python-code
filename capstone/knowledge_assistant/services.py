"""Wiring: builds the index and all services from Settings."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .agent import KnowledgeAgent, PendingStore
from pathlib import Path

from .config import ConfigError, Settings
from .documents import chunk_document, load_documents
from .embeddings import Embedder, build_embedder
from .llm import LLM, build_llm
from .rag import RagService
from .tools import ToolRegistry, build_registry
from .vector_store import IndexMismatchError, VectorStore

log = logging.getLogger("knowledge_assistant")


@dataclass
class Services:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    rag: RagService
    registry: ToolRegistry
    agent: KnowledgeAgent


def ingest(settings: Settings, embedder: Embedder) -> VectorStore:
    """Read documents, chunk them, embed them and save the index."""
    if not Path(settings.data_dir).is_dir():
        raise ConfigError(f"KB_DATA_DIR '{settings.data_dir}' does not exist. "
                          "Run commands from the capstone/ folder or use an absolute path.")
    documents = load_documents(settings.data_dir)
    if not documents:
        raise ConfigError(f"No .md or .txt documents with content were found in '{settings.data_dir}'.")
    chunks = [c for d in documents for c in chunk_document(d)]
    store = VectorStore(embedder.name)
    if chunks:
        store.add(chunks, embedder.embed([c.text for c in chunks]))
    store.save(settings.index_path)
    log.info("ingested documents=%d chunks=%d", len(documents), len(chunks))
    return store


def build_services(settings: Settings, llm: LLM | None = None, embedder: Embedder | None = None,
                   rebuild: bool = False) -> Services:
    embedder = embedder or build_embedder(settings)
    store = None
    if not rebuild and settings.index_path.exists():
        try:
            store = VectorStore.load(settings.index_path, embedder.name)
            if len(store) == 0:
                log.warning("saved index is empty; rebuilding")
                store = None
        except (IndexMismatchError, KeyError, ValueError) as exc:
            log.warning("index unusable (%s); rebuilding", type(exc).__name__)
    if store is None:
        store = ingest(settings, embedder)

    llm = llm or build_llm(settings)
    rag = RagService(store, embedder, llm, min_similarity=settings.min_similarity)
    registry = build_registry(rag, store, settings.notes_dir)
    agent = KnowledgeAgent(llm, registry, settings, PendingStore())
    return Services(settings, embedder, store, rag, registry, agent)
