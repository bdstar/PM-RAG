"""Cross-encoder reranking with a lexical fallback.

Primary: a sentence-transformers ``CrossEncoder`` (configurable).
Fallback: a lexical overlap score, used when the model is unavailable so the
pipeline stays functional offline. The active backend is reported.
"""
from __future__ import annotations

import re
from typing import Optional

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Reranker:
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._backend = None

    def _ensure(self):
        if self._backend is not None:
            return
        if not self.settings.reranker_enabled:
            self._backend = "disabled"
            return
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker '%s'...", self.settings.reranker_model)
            self._model = CrossEncoder(self.settings.reranker_model)
            self._backend = self.settings.reranker_model
        except Exception as exc:
            logger.warning("CrossEncoder unavailable (%s). Using lexical reranker.", exc)
            self._backend = "lexical-fallback"

    @property
    def backend(self) -> str:
        self._ensure()
        return self._backend

    def _lexical_score(self, query: str, doc: str) -> float:
        q = set(_TOKEN_RE.findall(query.lower()))
        d = set(_TOKEN_RE.findall(doc.lower()))
        if not q or not d:
            return 0.0
        return len(q & d) / len(q)

    def rerank(self, query: str, candidates: list[dict], top_k: Optional[int] = None) -> list[dict]:
        """candidates: list of dicts with at least 'document' and 'id'. Returns
        the same dicts with an added 'rerank_score', sorted descending."""
        self._ensure()
        if not candidates:
            return []
        if self._backend == "disabled":
            for c in candidates:
                c["rerank_score"] = c.get("fused_score", c.get("score", 0.0))
            ranked = candidates
        elif self._model is not None:
            pairs = [(query, c.get("document", "")) for c in candidates]
            scores = self._model.predict(pairs)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        else:  # lexical fallback
            for c in candidates:
                c["rerank_score"] = self._lexical_score(query, c.get("document", ""))
            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k] if top_k else ranked


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
