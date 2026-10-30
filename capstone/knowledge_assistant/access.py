"""Who may read what. Deny by default."""
from __future__ import annotations

from dataclasses import dataclass

# Higher number = more privileged. A role may read documents at or below its level.
LEVELS = {"public": 0, "student": 1, "staff": 2}


@dataclass(frozen=True)
class User:
    name: str
    role: str


def can_read(role: str, document_access: str) -> bool:
    """Unknown roles and unknown access labels are DENIED, never allowed."""
    if role not in LEVELS or document_access not in LEVELS:
        return False
    return LEVELS[role] >= LEVELS[document_access]
