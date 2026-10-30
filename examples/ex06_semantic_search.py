"""Project 3: semantic document search (Chapter 6).

Pipeline: load text files -> chunk -> embed -> store -> search.

Two embedders share one interface:

* HashingEmbedder           - offline, deterministic, LEXICAL (word overlap).
                              Good for learning and for automated tests.
* SentenceTransformerEmbedder - a real semantic model. Requires the optional
                              `sentence-transformers` package and a model
                              download. Not executed while this book was written.
"""
from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the to was were will with "
    "do does did i you we they this these those can how what when where which who why".split()
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_words(text: str, doc_id: str, source: str, max_words: int = 120,
                overlap_words: int = 20, metadata: dict | None = None) -> list[Chunk]:
    """Split text into overlapping windows of words.

    Overlap helps when an answer sits on a chunk boundary.
    """
    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be >= 0 and < max_words")
    words = text.split()
    chunks: list[Chunk] = []
    step = max_words - overlap_words
    for index, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start:start + max_words]
        if not window:
            break
        chunks.append(Chunk(f"{doc_id}#{index}", doc_id, source, " ".join(window), dict(metadata or {})))
        if start + max_words >= len(words):
            break
    return chunks


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


class HashingEmbedder:
    """Deterministic bag-of-words embedder using the hashing trick.

    It captures shared vocabulary, NOT meaning: "car" and "automobile" look
    unrelated to it. That limitation is the reason real embedding models exist.
    """

    def __init__(self, dim: int = 4096):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, bucket] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


class SentenceTransformerEmbedder:
    """Wrapper for a local Sentence Transformers model (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional import
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class SemanticIndex:
    """An in-memory vector index. Fine for thousands of chunks, not millions."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        new_vectors = self.embedder.embed([c.text for c in chunks])
        self._matrix = new_vectors if self._matrix is None else np.vstack([self._matrix, new_vectors])
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 3, min_score: float = 0.0,
               allow: Callable[[Chunk], bool] | None = None) -> list[SearchHit]:
        """Return the top-k chunks by cosine similarity.

        Vectors are unit length, so a dot product equals cosine similarity.
        `allow` filters BEFORE ranking, so forbidden chunks can never appear.
        """
        if self._matrix is None or not query.strip():
            return []
        query_vec = self.embedder.embed([query])[0]
        scores = self._matrix @ query_vec
        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for idx in order:
            chunk = self.chunks[int(idx)]
            if allow is not None and not allow(chunk):
                continue
            if scores[idx] < min_score:
                break
            hits.append(SearchHit(chunk, float(scores[idx])))
            if len(hits) == k:
                break
        return hits


def load_folder(folder: str | Path) -> list[Chunk]:
    """Chunk every .txt and .md file in a folder, keeping source metadata."""
    chunks: list[Chunk] = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        chunks.extend(chunk_words(text, doc_id=path.stem, source=path.name))
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit('Usage: python examples/ex06_semantic_search.py <folder> "your query"')
    index = SemanticIndex(HashingEmbedder())
    index.add(load_folder(sys.argv[1]))
    for hit in index.search(" ".join(sys.argv[2:]), k=3):
        print(f"{hit.score:.3f}  {hit.chunk.chunk_id}  {hit.chunk.text[:90]}...")
