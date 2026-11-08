"""BM25 keyword retrieval.

Uses ``rank_bm25`` if installed; otherwise a compact pure-Python BM25-Okapi
implementation (no extra dependency). The index is built from evidence-chunk
text held in the database and cached in memory; it is rebuilt when the vector
index version changes.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from ..core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class _PurePythonBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus_tokens
        self.n = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.df: Counter = Counter()
        self.tf: list[Counter] = []
        for doc in corpus_tokens:
            freqs = Counter(doc)
            self.tf.append(freqs)
            for term in freqs:
                self.df[term] += 1
        self.idf = {
            term: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in self.df.items()
        }

    def get_scores(self, query_tokens: list[str]):
        scores = [0.0] * self.n
        for term in query_tokens:
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i in range(self.n):
                freq = self.tf[i].get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * (freq * (self.k1 + 1)) / denom
        return scores


class BM25Index:
    def __init__(self):
        self.ids: list[str] = []
        self.id_to_pos: dict[str, int] = {}
        self._bm25 = None
        self._backend = None

    def build(self, docs: list[tuple[str, str]]):
        """docs: list of (chunk_key, text)."""
        self.ids = [d[0] for d in docs]
        self.id_to_pos = {cid: i for i, cid in enumerate(self.ids)}
        corpus_tokens = [tokenize(d[1]) for d in docs]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(corpus_tokens)
            self._backend = "rank_bm25"
        except Exception:
            self._bm25 = _PurePythonBM25(corpus_tokens)
            self._backend = "pure_python"
        logger.info("BM25 index built (%d docs, backend=%s)", len(self.ids), self._backend)

    @property
    def size(self) -> int:
        return len(self.ids)

    def search(self, query: str, top_k: int = 10, allowed_ids: Optional[set[str]] = None) -> list[dict]:
        if not self.ids or self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked:
            cid = self.ids[i]
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            if scores[i] <= 0:
                continue
            out.append({"id": cid, "score": float(scores[i])})
            if len(out) >= top_k:
                break
        return out
