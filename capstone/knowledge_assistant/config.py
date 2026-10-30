"""Configuration read from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .access import LEVELS, User


class ConfigError(ValueError):
    pass


def parse_api_keys(raw: str) -> dict[str, User]:
    """Parse 'key:user:role,key2:user2:role2' into {key: User}."""
    keys: dict[str, User] = {}
    for entry in filter(None, (part.strip() for part in raw.split(","))):
        pieces = entry.split(":")
        if len(pieces) != 3 or not all(pieces):
            raise ConfigError("KB_API_KEYS entries must look like key:user:role")
        key, name, role = pieces
        if role not in LEVELS:
            raise ConfigError(f"Unknown role '{role}'. Allowed: {sorted(LEVELS)}")
        keys[key] = User(name=name, role=role)
    return keys


@dataclass(frozen=True)
class Settings:
    embedder_backend: str = "hashing"
    data_dir: Path = Path("sample_data/docs")
    index_path: Path = Path(".kb_index.json")
    notes_dir: Path = Path("notes")
    min_similarity: float = 0.12
    max_iterations: int = 6
    timeout_seconds: float = 60.0
    llm_model: str = ""
    embedding_model: str = ""
    rate_limit_per_minute: int = 30
    api_keys: dict[str, User] = field(default_factory=dict, repr=False)  # repr=False: never print keys

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        try:
            return cls(
                embedder_backend=env.get("EMBEDDER_BACKEND", "hashing"),
                data_dir=Path(env.get("KB_DATA_DIR", "sample_data/docs")),
                index_path=Path(env.get("KB_INDEX_PATH", ".kb_index.json")),
                notes_dir=Path(env.get("KB_NOTES_DIR", "notes")),
                min_similarity=float(env.get("MIN_SIMILARITY", "0.12")),
                max_iterations=int(env.get("AGENT_MAX_ITERATIONS", "6")),
                timeout_seconds=float(env.get("AGENT_TIMEOUT_SECONDS", "60")),
                llm_model=env.get("LLM_MODEL", ""),
                embedding_model=env.get("EMBEDDING_MODEL", ""),
                rate_limit_per_minute=int(env.get("RATE_LIMIT_PER_MINUTE", "30")),
                api_keys=parse_api_keys(env.get("KB_API_KEYS", "")),
            )
        except ValueError as exc:
            if isinstance(exc, ConfigError):
                raise
            raise ConfigError(f"Invalid numeric setting: {exc}") from exc
