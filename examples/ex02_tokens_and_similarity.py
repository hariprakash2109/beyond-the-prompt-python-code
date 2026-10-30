"""Chapter 2 exercise: tokens, vectors and cosine similarity - no libraries.

This is a TOY. Real LLMs use learned subword tokenizers and learned
embeddings. The point here is to see the *mechanics* with numbers you can
check by hand.
"""
from __future__ import annotations

import math
import re


def toy_tokenize(text: str) -> list[str]:
    """Split text into lowercase words and punctuation marks."""
    return re.findall(r"[a-z0-9']+|[^\sa-z0-9]", text.lower())


def count_vector(tokens: list[str], vocabulary: list[str]) -> list[float]:
    """Represent a token list as word counts over a fixed vocabulary."""
    return [float(tokens.count(word)) for word in vocabulary]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two vectors (1.0 = same direction)."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    sentences = [
        "The cat sat on the mat.",
        "A cat is sitting on a mat.",
        "Quarterly revenue grew by ten percent.",
    ]
    token_lists = [toy_tokenize(s) for s in sentences]
    vocab = sorted({t for tokens in token_lists for t in tokens})
    vectors = [count_vector(t, vocab) for t in token_lists]
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            score = cosine_similarity(vectors[i], vectors[j])
            print(f"{i} vs {j}: {score:.3f}")
