"""Evidence indexing pipeline.

Materialises ``EvidenceDocument`` / ``EvidenceChunk`` rows from normalized DB
records and supplementary documents, embeds them, persists vectors in the
vector store, and records index/embedding versions. Supports incremental
indexing (only new chunks) and full rebuild. Prevents duplicate chunks via a
deterministic ``chunk_key`` and content hash.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.logging import get_logger
from ..database import models as m
from . import chunking
from .bm25 import BM25Index
from .embeddings import get_embedder
from .vector_store import get_vector_store

logger = get_logger(__name__)

ProgressCb = Optional[Callable[[int, int], None]]

_RECORD_SOURCES = {
    "citywide_budget_schedule": (m.ProjectBudgetSchedule, chunking.budget_schedule_to_text),
    "schedule_history": (m.ProjectScheduleHistory, chunking.schedule_history_to_text),
    "budget_spend_history": (m.ProjectBudgetHistory, chunking.budget_history_to_text),
}


def _chunk_key(dataset_type: str, record_id: int) -> str:
    return f"{dataset_type}:{record_id}"


def build_evidence_from_records(session: Session, limit_per_type: Optional[int] = None) -> dict:
    """Create EvidenceDocument/EvidenceChunk rows for records lacking them."""
    created = 0
    for dataset_type, (model, to_text) in _RECORD_SOURCES.items():
        dataset = session.scalar(select(m.Dataset).where(m.Dataset.dataset_type == dataset_type))
        if not dataset:
            continue
        existing_keys = set(
            session.scalars(
                select(m.EvidenceChunk.chunk_key).where(m.EvidenceChunk.dataset_type == dataset_type)
            ).all()
        )
        q = select(model)
        if limit_per_type:
            q = q.limit(limit_per_type)
        records = session.scalars(q).all()
        batch = 0
        for rec in records:
            key = _chunk_key(dataset_type, rec.id)
            if key in existing_keys:
                continue
            text = to_text(rec)
            doc_key = f"doc:{key}"
            doc = m.EvidenceDocument(
                document_key=doc_key,
                dataset_id=dataset.id,
                dataset_type=dataset_type,
                project_id=rec.project_id,
                document_type="structured_record",
                evidence_category="observed",
                source_file=None,
                source_record_id=str(rec.id),
                reporting_period=getattr(rec, "reporting_period", None),
                access_level="public",
                source_org="NYC Open Data",
            )
            session.add(doc)
            session.flush()
            chunk = m.EvidenceChunk(
                chunk_key=key,
                document_id=doc.id,
                source_id=str(rec.id),
                dataset_id=dataset.id,
                dataset_type=dataset_type,
                project_id=rec.project_id,
                fms_id=getattr(rec, "fms_id", None),
                record_id=str(rec.id),
                reporting_period=getattr(rec, "reporting_period", None),
                event_date=getattr(rec, "forecast_completion", None)
                or getattr(rec, "completion_date", None),
                document_type="structured_record",
                project_phase=getattr(rec, "current_phase", None),
                evidence_category="observed",
                original_text=text,
                normalized_text=text,
                content_hash=chunking.Chunk(key, text).content_hash,
                access_level="public",
                indexed=False,
            )
            session.add(chunk)
            created += 1
            batch += 1
            if batch >= 1000:
                session.commit()
                batch = 0
        session.commit()
    logger.info("Evidence build: %d new chunks", created)
    return {"chunks_created": created}


def index_pending_chunks(session: Session, batch_size: int = 256, progress_cb: ProgressCb = None) -> dict:
    """Embed and store all chunks not yet indexed."""
    settings = get_settings()
    embedder = get_embedder()
    store = get_vector_store()

    pending = session.scalars(select(m.EvidenceChunk).where(m.EvidenceChunk.indexed.is_(False))).all()
    total = len(pending)
    if total == 0:
        return {"indexed": 0, "embed_backend": embedder.backend, "vector_backend": store.backend}

    # Index version record.
    version = m.RetrievalIndexVersion(
        version_key=uuid.uuid4().hex[:12],
        embed_model=embedder.backend,
        dim=embedder.dim,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        collection_name=settings.collection_name,
    )
    session.add(version)
    session.flush()

    done = 0
    for start in range(0, total, batch_size):
        batch = pending[start:start + batch_size]
        texts = [c.normalized_text or c.original_text for c in batch]
        embeddings = embedder.encode(texts)
        ids = [c.chunk_key for c in batch]
        metadatas = [_chunk_metadata(c) for c in batch]
        store.add(ids, embeddings, metadatas, texts)
        for c, emb in zip(batch, embeddings):
            c.indexed = True
            session.add(
                m.EmbeddingMetadata(
                    chunk_id=c.id, index_version_id=version.id,
                    embed_model=embedder.backend, dim=embedder.dim, vector_id=c.chunk_key,
                )
            )
        done += len(batch)
        session.commit()
        if progress_cb:
            progress_cb(done, total)

    version.chunk_count = store.count()
    version.dim = embedder.dim
    session.commit()
    logger.info("Indexed %d chunks (embed=%s, store=%s)", done, embedder.backend, store.backend)
    return {"indexed": done, "embed_backend": embedder.backend,
            "vector_backend": store.backend, "index_version": version.version_key}


def _chunk_metadata(c: m.EvidenceChunk) -> dict:
    return {
        "chunk_key": c.chunk_key,
        "document_id": c.document_id,
        "source_id": c.source_id or "",
        "dataset_id": c.dataset_id or 0,
        "dataset_type": c.dataset_type or "",
        "project_id": c.project_id or 0,
        "fms_id": c.fms_id or "",
        "record_id": c.record_id or "",
        "reporting_period": c.reporting_period or "",
        "document_type": c.document_type or "",
        "project_phase": c.project_phase or "",
        "evidence_category": c.evidence_category or "",
        "access_level": c.access_level or "public",
    }


def index_supplementary_documents(session: Session) -> dict:
    """Index PDF/TXT/MD files from the knowledge directory as evidence."""
    settings = get_settings()
    kdir = settings.knowledge_path
    if not kdir.exists():
        return {"documents": 0, "chunks": 0}
    created_docs = created_chunks = 0
    for path in kdir.glob("**/*"):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        doc_key = f"supp:{path.name}"
        if session.scalar(select(m.EvidenceDocument).where(m.EvidenceDocument.document_key == doc_key)):
            continue
        text = _read_document(path)
        if not text.strip():
            continue
        doc = m.EvidenceDocument(
            document_key=doc_key,
            dataset_type="supplementary",
            document_type="guidance_or_lessons",
            evidence_category="guidance",
            source_file=path.name,
            title=path.stem,
            access_level="public",
            source_org="Supplementary knowledge source",
        )
        session.add(doc)
        session.flush()
        created_docs += 1
        for i, piece in enumerate(chunking.sentence_chunks(text)):
            key = f"{doc_key}:{i}"
            session.add(
                m.EvidenceChunk(
                    chunk_key=key, document_id=doc.id, source_id=path.name,
                    dataset_type="supplementary", document_type="guidance_or_lessons",
                    evidence_category="guidance", source_file=path.name,
                    original_text=piece, normalized_text=piece,
                    content_hash=chunking.Chunk(key, piece).content_hash,
                    access_level="public", indexed=False,
                )
            )
            created_chunks += 1
        session.commit()
    return {"documents": created_docs, "chunks": created_chunks}


def _read_document(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        logger.warning("Could not read PDF %s: %s", path, exc)
        return ""


# ---------------------------------------------------------------------------
# BM25 in-memory index (rebuilt from DB chunks)
# ---------------------------------------------------------------------------
_bm25: BM25Index | None = None
_bm25_signature: tuple | None = None


def get_bm25_index(session: Session, force: bool = False) -> BM25Index:
    global _bm25, _bm25_signature
    count = session.scalar(select(func.count(m.EvidenceChunk.id))) or 0
    signature = (count,)
    if _bm25 is None or force or signature != _bm25_signature:
        docs = session.execute(
            select(m.EvidenceChunk.chunk_key, m.EvidenceChunk.normalized_text)
        ).all()
        idx = BM25Index()
        idx.build([(k, t or "") for k, t in docs])
        _bm25 = idx
        _bm25_signature = signature
    return _bm25


def full_reindex(session: Session, limit_per_type: Optional[int] = None,
                 rebuild_vectors: bool = True, progress_cb: ProgressCb = None) -> dict:
    """Full evidence build + (re)index."""
    if rebuild_vectors:
        get_vector_store().reset()
        session.query(m.EvidenceChunk).update({m.EvidenceChunk.indexed: False})
        session.commit()
    stats_build = build_evidence_from_records(session, limit_per_type=limit_per_type)
    stats_supp = index_supplementary_documents(session)
    stats_index = index_pending_chunks(session, progress_cb=progress_cb)
    get_bm25_index(session, force=True)
    return {**stats_build, "supplementary": stats_supp, **stats_index}
