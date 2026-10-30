"""A small in-memory vector store with JSON persistence and access filtering."""
from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .access import User, can_read
from .documents import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class IndexMismatchError(Exception):
    """The saved index was built with a different embedder than the current one."""


class VectorStore:
    def __init__(self, embedder_name: str):
        self.embedder_name = embedder_name
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)

    def get_document_chunks(self, doc_id: str) -> list[Chunk]:
        return [c for c in self.chunks if c.doc_id == doc_id]

    def search(self, query_vector: np.ndarray, user: User, k: int = 4,
               min_score: float = 0.0) -> list[Hit]:
        """Top-k chunks the USER IS ALLOWED TO READ, best first.

        Unauthorized chunks are removed before ranking, so they cannot appear
        in the results, in the prompt, or in an error message.
        """
        if self.matrix is None or k <= 0:
            return []
        allowed = [i for i, c in enumerate(self.chunks) if can_read(user.role, c.access)]
        if not allowed:
            return []
        scores = self.matrix[allowed] @ query_vector
        ranked = np.argsort(-scores)[:k]
        return [Hit(self.chunks[allowed[int(r)]], float(scores[int(r)]))
                for r in ranked if scores[int(r)] >= min_score]

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedder": self.embedder_name,
            "chunks": [asdict(c) for c in self.chunks],
            "matrix": None if self.matrix is None else {
                "shape": list(self.matrix.shape),
                "data": base64.b64encode(self.matrix.astype(np.float32).tobytes()).decode("ascii"),
            },
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, expected_embedder: str) -> "VectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["embedder"] != expected_embedder:
            raise IndexMismatchError(
                f"Index was built with '{payload['embedder']}' but the current embedder is "
                f"'{expected_embedder}'. Rebuild the index.")
        store = cls(expected_embedder)
        store.chunks = [Chunk(**c) for c in payload["chunks"]]
        if payload["matrix"] is not None:
            raw = base64.b64decode(payload["matrix"]["data"])
            store.matrix = np.frombuffer(raw, dtype=np.float32).reshape(payload["matrix"]["shape"]).copy()
        return store
