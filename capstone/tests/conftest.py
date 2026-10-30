import dataclasses

import pytest

from helpers import SAMPLE_DOCS, STAFF, STUDENT
from knowledge_assistant.config import Settings
from knowledge_assistant.embeddings import HashingEmbedder
from knowledge_assistant.llm import ScriptedLLM
from knowledge_assistant.services import build_services


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=SAMPLE_DOCS, index_path=tmp_path / "index.json", notes_dir=tmp_path / "notes",
        max_iterations=4, timeout_seconds=30, rate_limit_per_minute=100,
        api_keys={"k-student": STUDENT, "k-staff": STAFF},
    )


@pytest.fixture
def make_services(settings):
    def factory(replies=None, **overrides):
        llm = ScriptedLLM(replies or [])
        cfg = dataclasses.replace(settings, **overrides) if overrides else settings
        return build_services(cfg, llm=llm, embedder=HashingEmbedder(), rebuild=True), llm
    return factory
