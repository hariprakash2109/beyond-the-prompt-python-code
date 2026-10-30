"""Embedding back-ends behind one small interface.

hashing               offline, deterministic, lexical. Used for tests and demos.
openai                OpenAI embeddings API (needs a key and a model name).
sentence-transformers local model (optional dependency).

The OpenAI and Sentence Transformers back-ends were NOT executed while this
book was written; the automated tests use the hashing embedder.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol

import numpy as np

from .config import ConfigError, Settings

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the to was were will with "
    "do does did i you we they this these those can how what when where which who why".split()
)


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of unit-length vectors."""
        ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


class HashingEmbedder:
    def __init__(self, dim: int = 4096):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _tokens(self, text: str) -> list[str]:
        words = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                matrix[row, bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
        return _normalize(matrix)


class OpenAIEmbedder:
    def __init__(self, model: str, client: Any | None = None, batch_size: int = 64):
        if not model:
            raise ConfigError("EMBEDDING_MODEL is required when EMBEDDER_BACKEND=openai")
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.model, self.client, self.batch_size = model, client, batch_size
        self.name = f"openai-{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return _normalize(np.array(vectors))


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError("Install the optional package: pip install sentence-transformers") from exc
        self.model = SentenceTransformer(model_name)
        self.name = f"st-{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.model.encode(texts))


def build_embedder(settings: Settings) -> Embedder:
    backend = settings.embedder_backend
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "openai":
        return OpenAIEmbedder(settings.embedding_model)
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model or "all-MiniLM-L6-v2")
    raise ConfigError(f"Unknown EMBEDDER_BACKEND '{backend}'")
