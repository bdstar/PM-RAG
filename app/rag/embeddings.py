"""Embedding model abstraction with a deterministic offline fallback.

Primary: Hugging Face sentence-transformers (configurable model).
Fallback: a deterministic hashing / bag-of-words embedder that requires no model
download, so ingestion, indexing and retrieval remain fully functional offline
and in CI. The active backend is reported so the UI/experiments can record it.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import List

import numpy as np

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _HashingEmbedder:
    """Deterministic hashing embedder (TF over hashed tokens, L2-normalised).

    Not as strong as a trained model, but stable, dependency-free, and good
    enough to make dense retrieval meaningful in offline/test environments.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.name = f"hashing-fallback-{dim}d"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN_RE.findall((text or "").lower())
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 1) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]) if texts else np.zeros((0, self.dim))


class Embedder:
    """Lazy wrapper that prefers sentence-transformers, falls back to hashing."""

    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._backend = None
        self._dim = None

    def _ensure(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s'...", self.settings.embed_model)
            self._model = SentenceTransformer(self.settings.embed_model)
            self._dim = self._model.get_sentence_embedding_dimension()
            self._backend = self.settings.embed_model
            logger.info("Embedding backend: %s (dim=%d)", self._backend, self._dim)
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable (%s). Using deterministic "
                "hashing embedder fallback.", exc,
            )
            self._model = _HashingEmbedder(self.settings.embed_fallback_dim)
            self._dim = self.settings.embed_fallback_dim
            self._backend = self._model.name

    @property
    def dim(self) -> int:
        self._ensure()
        return self._dim

    @property
    def backend(self) -> str:
        self._ensure()
        return self._backend

    @property
    def is_fallback(self) -> bool:
        self._ensure()
        return isinstance(self._model, _HashingEmbedder)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._ensure()
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        if isinstance(self._model, _HashingEmbedder):
            return self._model.encode(texts)
        emb = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
