"""Hybrid retrieval: BM25 + dense fusion (RRF) + cross-encoder reranking.

Implements the four retrieval modes and enforces temporal validity (only
evidence available at or before the scenario's as-of date is retrievable).
Metadata filtering is applied for mode/dataset/evidence-category. The returned
similarity scores are retrieval scores only and MUST NOT be presented as
calibrated factual-confidence probabilities (spec 10).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.security import sanitize_evidence_text
from ..database import models as m
from . import indexer
from .embeddings import get_embedder
from .reranker import get_reranker
from .vector_store import get_vector_store

logger = get_logger(__name__)

RETRIEVAL_MODES = [
    "current_project",
    "same_project_historical",
    "cross_project",
    "hybrid_temporal",
]


@dataclass
class RetrievalParams:
    mode: str = "hybrid_temporal"
    project_id: Optional[int] = None
    as_of_period: Optional[str] = None  # YYYYMM
    top_k: int = 8
    candidate_k: int = 40
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    rrf_k: int = 60
    similarity_threshold: float = 0.0
    reranker_enabled: bool = True
    dataset_type: Optional[str] = None
    evidence_category: Optional[str] = None
    enabled_retrievers: tuple = ("bm25", "dense")


@dataclass
class EvidenceItem:
    chunk_key: str
    chunk_id: Optional[int]
    text: str
    metadata: dict
    bm25_score: float = 0.0
    dense_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0

    def to_dict(self) -> dict:
        md = self.metadata or {}
        return {
            "chunk_key": self.chunk_key,
            "chunk_id": self.chunk_id,
            "source_id": md.get("source_id"),
            "dataset_type": md.get("dataset_type"),
            "project_id": md.get("project_id"),
            "reporting_period": md.get("reporting_period"),
            "evidence_category": md.get("evidence_category"),
            "document_type": md.get("document_type"),
            "text": self.text,
            "bm25_score": round(self.bm25_score, 4),
            "dense_score": round(self.dense_score, 4),
            "fused_score": round(self.fused_score, 6),
            "rerank_score": round(self.rerank_score, 4),
            "rank": self.rank,
        }


def _latest_period_for_project(session: Session, project_id: int, as_of: Optional[str]) -> Optional[str]:
    q = select(m.EvidenceChunk.reporting_period).where(
        m.EvidenceChunk.project_id == project_id,
        m.EvidenceChunk.reporting_period != "",
        m.EvidenceChunk.reporting_period.is_not(None),
    )
    if as_of:
        q = q.where(m.EvidenceChunk.reporting_period <= as_of)
    periods = [p for p in session.scalars(q).all() if p]
    return max(periods) if periods else None


def _allowed_chunk_keys(session: Session, params: RetrievalParams) -> set[str]:
    """Compute the set of chunk_keys allowed under mode + temporal + filters."""
    q = select(m.EvidenceChunk.chunk_key, m.EvidenceChunk.reporting_period,
               m.EvidenceChunk.project_id, m.EvidenceChunk.dataset_type,
               m.EvidenceChunk.evidence_category)
    conds = [m.EvidenceChunk.access_level == "public"]

    mode = params.mode
    if mode in ("current_project", "same_project_historical") and params.project_id:
        conds.append(m.EvidenceChunk.project_id == params.project_id)
        if mode == "current_project":
            current = _latest_period_for_project(session, params.project_id, params.as_of_period)
            if current:
                conds.append(m.EvidenceChunk.reporting_period == current)
    elif mode == "cross_project" and params.project_id:
        conds.append(m.EvidenceChunk.project_id != params.project_id)

    if params.dataset_type:
        conds.append(m.EvidenceChunk.dataset_type == params.dataset_type)
    if params.evidence_category:
        conds.append(m.EvidenceChunk.evidence_category == params.evidence_category)

    # Temporal validity: reporting_period <= as_of, OR undated supplementary.
    if params.as_of_period and mode != "current_project":
        conds.append(
            or_(
                m.EvidenceChunk.reporting_period <= params.as_of_period,
                m.EvidenceChunk.reporting_period == "",
                m.EvidenceChunk.reporting_period.is_(None),
            )
        )

    rows = session.execute(q.where(*conds)).all()
    return {r[0] for r in rows}


def _meta_ok(md: dict, params: RetrievalParams, allowed: set[str]) -> bool:
    return md.get("chunk_key") in allowed


def retrieve(session: Session, query: str, params: RetrievalParams) -> list[EvidenceItem]:
    settings = get_settings()
    allowed = _allowed_chunk_keys(session, params)
    if not allowed:
        return []

    results: dict[str, EvidenceItem] = {}
    bm25_ranked: list[str] = []
    dense_ranked: list[str] = []

    # --- BM25 ---
    if "bm25" in params.enabled_retrievers:
        bm25 = indexer.get_bm25_index(session)
        hits = bm25.search(query, top_k=params.candidate_k, allowed_ids=allowed)
        for h in hits:
            bm25_ranked.append(h["id"])
            item = results.setdefault(h["id"], EvidenceItem(h["id"], None, "", {}))
            item.bm25_score = h["score"]

    # --- Dense ---
    if "dense" in params.enabled_retrievers:
        embedder = get_embedder()
        store = get_vector_store()
        qvec = embedder.encode_one(query)
        where = None
        if params.mode in ("current_project", "same_project_historical") and params.project_id:
            where = {"project_id": params.project_id}
        raw = store.query(qvec, top_k=max(params.candidate_k * 4, 80), where=where)
        dense_hits = [r for r in raw if _meta_ok(r.get("metadata", {}), params, allowed)]
        dense_hits = dense_hits[: params.candidate_k]
        for r in dense_hits:
            cid = r["id"]
            dense_ranked.append(cid)
            item = results.setdefault(cid, EvidenceItem(cid, None, r.get("document", ""), r.get("metadata", {})))
            item.dense_score = r["score"]
            if not item.text:
                item.text = r.get("document", "")
            if not item.metadata:
                item.metadata = r.get("metadata", {})

    if not results:
        return []

    # --- Weighted Reciprocal Rank Fusion ---
    rrf_k = params.rrf_k
    bm25_pos = {cid: i for i, cid in enumerate(bm25_ranked)}
    dense_pos = {cid: i for i, cid in enumerate(dense_ranked)}
    for cid, item in results.items():
        score = 0.0
        if cid in bm25_pos:
            score += params.bm25_weight * (1.0 / (rrf_k + bm25_pos[cid] + 1))
        if cid in dense_pos:
            score += params.dense_weight * (1.0 / (rrf_k + dense_pos[cid] + 1))
        item.fused_score = score

    fused = sorted(results.values(), key=lambda x: x.fused_score, reverse=True)
    fused = fused[: params.candidate_k]

    # --- Hydrate text/metadata/db-id from DB for anything missing (e.g. BM25-only) ---
    missing_keys = [i.chunk_key for i in fused if not i.text or i.chunk_id is None]
    if missing_keys:
        rows = session.scalars(
            select(m.EvidenceChunk).where(m.EvidenceChunk.chunk_key.in_(missing_keys))
        ).all()
        by_key = {r.chunk_key: r for r in rows}
        for item in fused:
            rec = by_key.get(item.chunk_key)
            if rec:
                item.chunk_id = rec.id
                if not item.text:
                    item.text = rec.normalized_text or rec.original_text
                if not item.metadata:
                    item.metadata = indexer._chunk_metadata(rec)
                item.metadata.setdefault("chunk_key", rec.chunk_key)

    # Ensure chunk_id populated for all.
    unresolved = [i.chunk_key for i in fused if i.chunk_id is None]
    if unresolved:
        rows = session.scalars(
            select(m.EvidenceChunk).where(m.EvidenceChunk.chunk_key.in_(unresolved))
        ).all()
        by_key = {r.chunk_key: r for r in rows}
        for item in fused:
            if item.chunk_id is None and item.chunk_key in by_key:
                item.chunk_id = by_key[item.chunk_key].id

    # Sanitize evidence text (prompt-injection defence) before it reaches the LLM.
    for item in fused:
        item.text = sanitize_evidence_text(item.text)

    # --- Rerank ---
    if params.reranker_enabled:
        reranker = get_reranker()
        candidates = [{"id": i.chunk_key, "document": i.text, "fused_score": i.fused_score,
                       "_item": i} for i in fused]
        ranked = reranker.rerank(query, candidates, top_k=params.top_k)
        final = []
        for rank, c in enumerate(ranked, 1):
            item = c["_item"]
            item.rerank_score = c.get("rerank_score", 0.0)
            item.rank = rank
            final.append(item)
    else:
        final = fused[: params.top_k]
        for rank, item in enumerate(final, 1):
            item.rank = rank
            item.rerank_score = item.fused_score

    # Similarity threshold (on dense score) filter, keeping at least nothing fabricated.
    if params.similarity_threshold > 0:
        final = [i for i in final if i.dense_score >= params.similarity_threshold or i.bm25_score > 0]

    return final
