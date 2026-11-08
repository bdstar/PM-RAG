"""Lightweight security helpers.

This research platform is single-tenant and local, but the research spec
requires (a) access-level handling for restricted documents, (b) treating
retrieved content as untrusted (prompt-injection defence), and (c) never
hard-coding credentials. These helpers implement (a) and (b); credential
handling lives in ``core.config`` (environment only).
"""
from __future__ import annotations

import re

# Ordered from least to most restrictive.
ACCESS_LEVELS = ["public", "internal", "restricted"]

# Phrases commonly used in prompt-injection attempts embedded in documents.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(above|previous|prior)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"</?(system|assistant|user)>", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
]


def can_access(user_level: str, doc_level: str) -> bool:
    """Return True if a user at ``user_level`` may read a ``doc_level`` doc."""
    try:
        return ACCESS_LEVELS.index(user_level) >= ACCESS_LEVELS.index(doc_level)
    except ValueError:
        # Unknown levels default to the most restrictive policy.
        return user_level == doc_level == "public"


def sanitize_evidence_text(text: str) -> str:
    """Neutralise embedded prompt-injection instructions in retrieved text.

    The text is NOT trusted as instructions. We redact known injection phrases
    so they cannot steer the generator, while preserving the factual content for
    citation/audit.
    """
    if not text:
        return text
    cleaned = text
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[redacted-instruction]", cleaned)
    return cleaned


def detect_injection(text: str) -> bool:
    """Return True if the text contains likely prompt-injection content."""
    return any(pat.search(text or "") for pat in _INJECTION_PATTERNS)
