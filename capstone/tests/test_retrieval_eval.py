"""Retrieval-quality tests using the golden question set (hypothetical data)."""
import json

from helpers import EVAL_FILE
from knowledge_assistant.access import User
from knowledge_assistant.embeddings import HashingEmbedder
from knowledge_assistant.services import ingest

EVAL = json.loads(EVAL_FILE.read_text())


def _retrieve(store, question, role, k=3, min_score=0.0):
    vec = HashingEmbedder().embed([question])[0]
    return store.search(vec, User("eval", role), k=k, min_score=min_score)


def test_recall_at_3_on_answerable_questions(settings):
    store = ingest(settings, HashingEmbedder())
    hits_found = 0
    for item in EVAL["answerable"]:
        docs = [h.chunk.doc_id for h in _retrieve(store, item["question"], item["role"])]
        hits_found += item["expected_doc"] in docs
    recall = hits_found / len(EVAL["answerable"])
    assert recall >= 0.9, f"recall@3 = {recall:.2f}"


def test_top1_precision_on_answerable_questions(settings):
    store = ingest(settings, HashingEmbedder())
    correct = sum(
        _retrieve(store, i["question"], i["role"], k=1)[0].chunk.doc_id == i["expected_doc"]
        for i in EVAL["answerable"])
    assert correct / len(EVAL["answerable"]) >= 0.8


def test_unanswerable_questions_fall_below_threshold(settings):
    store = ingest(settings, HashingEmbedder())
    for item in EVAL["unanswerable"]:
        assert _retrieve(store, item["question"], item["role"], min_score=settings.min_similarity) == [], item


def test_restricted_documents_never_leak_to_lower_roles(settings):
    store = ingest(settings, HashingEmbedder())
    for item in EVAL["must_not_leak"]:
        docs = {h.chunk.doc_id for h in _retrieve(store, item["question"], item["role"], k=50)}
        assert item["forbidden_doc"] not in docs, item
