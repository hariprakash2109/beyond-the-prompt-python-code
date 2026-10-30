"""Loading documents and splitting them into chunks that keep their metadata."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .access import LEVELS

SUPPORTED_SUFFIXES = {".md", ".txt"}
DEFAULT_ACCESS = "staff"  # fail closed: a document with no label is treated as restricted


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    access: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    access: str
    text: str
    position: int


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Read an optional '---' block of 'key: value' lines at the top of a file."""
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    meta: dict[str, str] = {}
    for end in range(1, len(lines)):
        if lines[end].strip() == "---":
            body = "\n".join(lines[end + 1:]).strip()
            return meta, body
        if ":" in lines[end]:
            key, value = lines[end].split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return {}, raw  # no closing marker: treat the whole file as body


def load_documents(folder: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        access = meta.get("access", DEFAULT_ACCESS).lower()
        if access not in LEVELS:
            access = DEFAULT_ACCESS  # unknown label -> most restrictive
        if not body.strip():
            continue
        documents.append(Document(
            doc_id=path.stem, title=meta.get("title", path.stem.replace("_", " ").title()),
            access=access, source=path.name, text=body))
    return documents


def chunk_document(doc: Document, max_words: int = 120, overlap_words: int = 20) -> list[Chunk]:
    if max_words <= 0 or not 0 <= overlap_words < max_words:
        raise ValueError("Require max_words > 0 and 0 <= overlap_words < max_words")
    words = doc.text.split()
    chunks: list[Chunk] = []
    step = max_words - overlap_words
    for position, start in enumerate(range(0, max(len(words), 1), step)):
        window = words[start:start + max_words]
        if not window:
            break
        chunks.append(Chunk(f"{doc.doc_id}#{position}", doc.doc_id, doc.title, doc.source,
                            doc.access, " ".join(window), position))
        if start + max_words >= len(words):
            break
    return chunks
