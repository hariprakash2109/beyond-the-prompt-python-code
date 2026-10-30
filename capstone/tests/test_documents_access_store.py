import numpy as np
import pytest

from helpers import STAFF, STUDENT
from knowledge_assistant.access import User, can_read
from knowledge_assistant.config import ConfigError, parse_api_keys
from knowledge_assistant.documents import chunk_document, load_documents, parse_front_matter
from knowledge_assistant.embeddings import HashingEmbedder
from knowledge_assistant.vector_store import IndexMismatchError, VectorStore


def test_front_matter_parsing():
    meta, body = parse_front_matter("---\ntitle: T\naccess: staff\n---\nHello body")
    assert meta == {"title": "T", "access": "staff"} and body == "Hello body"
    assert parse_front_matter("no front matter") == ({}, "no front matter")
    assert parse_front_matter("---\ntitle: never closed")[0] == {}


@pytest.mark.parametrize("role,access,expected", [
    ("staff", "staff", True), ("staff", "public", True), ("student", "staff", False),
    ("student", "student", True), ("public", "student", False),
    ("hacker", "public", False), ("staff", "top-secret", False),
])
def test_can_read_matrix_denies_unknowns(role, access, expected):
    assert can_read(role, access) is expected


def test_missing_or_unknown_access_label_fails_closed(tmp_path):
    (tmp_path / "a.md").write_text("no label here, just words about topics")
    (tmp_path / "b.md").write_text("---\naccess: everyone\n---\nunknown label text words")
    docs = {d.doc_id: d for d in load_documents(tmp_path)}
    assert docs["a"].access == "staff" and docs["b"].access == "staff"


def test_loader_skips_unsupported_and_empty_files(tmp_path):
    (tmp_path / "x.pdf").write_text("ignored")
    (tmp_path / "empty.md").write_text("---\naccess: public\n---\n")
    assert load_documents(tmp_path) == []


def test_chunks_keep_metadata_and_validate_args(tmp_path):
    (tmp_path / "d.md").write_text("---\ntitle: Doc\naccess: student\n---\n" + " ".join(["word"] * 300))
    doc = load_documents(tmp_path)[0]
    chunks = chunk_document(doc, max_words=100, overlap_words=10)
    assert len(chunks) >= 3
    assert all(c.access == "student" and c.title == "Doc" and c.doc_id == "d" for c in chunks)
    with pytest.raises(ValueError):
        chunk_document(doc, max_words=10, overlap_words=10)


def test_parse_api_keys():
    keys = parse_api_keys("k1:alice:student, k2:bob:staff")
    assert keys["k2"] == User("bob", "staff")
    with pytest.raises(ConfigError):
        parse_api_keys("bad-entry")
    with pytest.raises(ConfigError):
        parse_api_keys("k:u:superadmin")


def _store(settings):
    from knowledge_assistant.services import ingest
    return ingest(settings, HashingEmbedder())


def test_search_never_returns_unauthorized_chunks(settings):
    store, emb = _store(settings), HashingEmbedder()
    q = emb.embed(["What placement fee do partner employers pay?"])[0]
    student_docs = {h.chunk.doc_id for h in store.search(q, STUDENT, k=20)}
    staff_docs = {h.chunk.doc_id for h in store.search(q, STAFF, k=20)}
    assert "employer_partner_contracts" not in student_docs
    assert "employer_partner_contracts" in staff_docs


def test_unknown_role_sees_nothing(settings):
    store, emb = _store(settings), HashingEmbedder()
    assert store.search(emb.embed(["internship"])[0], User("x", "intruder"), k=5) == []


def test_index_roundtrip_and_embedder_mismatch(settings):
    store = _store(settings)
    loaded = VectorStore.load(settings.index_path, "hashing-4096")
    assert len(loaded) == len(store)
    assert np.allclose(loaded.matrix, store.matrix)
    with pytest.raises(IndexMismatchError):
        VectorStore.load(settings.index_path, "some-other-embedder")


def test_add_rejects_mismatched_lengths():
    store = VectorStore("x")
    with pytest.raises(ValueError):
        store.add([object()], np.zeros((2, 3), dtype=np.float32))


def test_ingest_fails_loudly_on_missing_or_empty_data_dir(settings, tmp_path):
    import dataclasses
    from knowledge_assistant.config import ConfigError
    from knowledge_assistant.services import ingest
    with pytest.raises(ConfigError, match="does not exist"):
        ingest(dataclasses.replace(settings, data_dir=tmp_path / "nope"), HashingEmbedder())
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ConfigError, match="No .md or .txt"):
        ingest(dataclasses.replace(settings, data_dir=empty), HashingEmbedder())


def test_an_empty_saved_index_is_rebuilt_not_trusted(settings):
    from knowledge_assistant.llm import ScriptedLLM
    from knowledge_assistant.services import build_services
    VectorStore("hashing-4096").save(settings.index_path)          # simulate the bad index
    services = build_services(settings, llm=ScriptedLLM([]), embedder=HashingEmbedder())
    assert len(services.store) > 0
