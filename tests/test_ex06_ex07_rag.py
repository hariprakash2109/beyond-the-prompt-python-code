import pytest

from ex06_semantic_search import Chunk, HashingEmbedder, SemanticIndex, chunk_words
from ex07_rag_assistant import NO_ANSWER, answer_question

DOCS = {
    "refunds": "Refunds are issued within 14 days of purchase when the receipt is provided. "
               "Digital goods cannot be refunded after download.",
    "laptops": "Staff laptops are replaced every three years. Report a broken laptop to the IT desk.",
    "holidays": "The office is closed on public holidays and between December 25 and January 1.",
}


def make_index():
    index = SemanticIndex(HashingEmbedder())
    for name, text in DOCS.items():
        index.add(chunk_words(text, doc_id=name, source=f"{name}.txt"))
    return index


def test_chunking_overlap_and_ids():
    text = " ".join(f"w{i}" for i in range(50))
    chunks = chunk_words(text, "d", "d.txt", max_words=20, overlap_words=5)
    assert chunks[0].chunk_id == "d#0"
    assert chunks[0].text.split()[-5:] == chunks[1].text.split()[:5]
    assert all(c.doc_id == "d" and c.source == "d.txt" for c in chunks)


def test_chunking_validates_arguments():
    with pytest.raises(ValueError):
        chunk_words("a b c", "d", "d", max_words=5, overlap_words=5)


def test_embeddings_are_unit_length_and_deterministic():
    emb = HashingEmbedder()
    a = emb.embed(["hello world"])
    b = emb.embed(["hello world"])
    assert (a == b).all()
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5


def test_search_finds_relevant_document_first():
    hits = make_index().search("How long do refunds take?", k=2)
    assert hits[0].chunk.doc_id == "refunds"


def test_search_filter_blocks_unauthorized_chunks_before_ranking():
    hits = make_index().search("refunds receipt", k=3, allow=lambda c: c.doc_id != "refunds")
    assert all(h.chunk.doc_id != "refunds" for h in hits)


def test_empty_query_returns_nothing():
    assert make_index().search("   ") == []


def test_rag_answers_with_valid_citation():
    result = answer_question("How long do refunds take?", make_index(),
                             lambda prompt: "Within 14 days with a receipt [1].")
    assert not result.abstained
    assert result.citations and result.citations[0].startswith("refunds")


def test_rag_abstains_without_calling_model_when_nothing_relevant():
    def explode(prompt):
        raise AssertionError("model must not be called")
    result = answer_question("What is the capital of Mongolia?", make_index(), explode)
    assert result.abstained and result.answer == NO_ANSWER


def test_rag_rejects_invented_citation():
    result = answer_question("How long do refunds take?", make_index(),
                             lambda prompt: "Fourteen days [9].")
    assert result.abstained


def test_rag_rejects_uncited_answer():
    result = answer_question("How long do refunds take?", make_index(),
                             lambda prompt: "Fourteen days.")
    assert result.abstained


def test_rag_prompt_contains_injection_warning_and_sources():
    seen = {}

    def spy(prompt):
        seen["p"] = prompt
        return "Within 14 days [1]."
    answer_question("refund receipt days", make_index(), spy)
    assert "untrusted data" in seen["p"] and "[1]" in seen["p"]
