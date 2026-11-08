"""Persistent vector store with ChromaDB and a numpy on-disk fallback.

ChromaDB is the default persistent store (spec). If chromadb is not installed,
a numpy-backed store persisted to disk is used so dense retrieval still works.
Both stores support metadata filtering by dataset/project/period/category.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class _NumpyStore:
    """Simple persistent cosine-similarity store (fallback)."""

    def __init__(self, path: Path, name: str):
        self.path = path
        self.name = name
        self.file = path / f"{name}.pkl"
        self.ids: list[str] = []
        self.vectors: Optional[np.ndarray] = None
        self.metadatas: list[dict] = []
        self.documents: list[str] = []
        self._load()

    def _load(self):
        if self.file.exists():
            with open(self.file, "rb") as f:
                data = pickle.load(f)
            self.ids = data["ids"]
            self.vectors = data["vectors"]
            self.metadatas = data["metadatas"]
            self.documents = data["documents"]

    def _save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.file, "wb") as f:
            pickle.dump(
                {"ids": self.ids, "vectors": self.vectors,
                 "metadatas": self.metadatas, "documents": self.documents}, f,
            )

    def add(self, ids, embeddings, metadatas, documents):
        idset = set(self.ids)
        new = [(i, e, m, d) for i, e, m, d in zip(ids, embeddings, metadatas, documents) if i not in idset]
        if not new:
            return
        vs = np.asarray([e for _, e, _, _ in new], dtype=np.float32)
        self.ids.extend([i for i, _, _, _ in new])
        self.metadatas.extend([m for _, _, m, _ in new])
        self.documents.extend([d for _, _, _, d in new])
        self.vectors = vs if self.vectors is None else np.vstack([self.vectors, vs])
        self._save()

    def _match(self, meta: dict, where: dict) -> bool:
        for k, v in where.items():
            if meta.get(k) != v:
                return False
        return True

    def query(self, embedding, top_k=10, where=None):
        if self.vectors is None or len(self.ids) == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32)
        qn = q / (np.linalg.norm(q) or 1.0)
        sims = self.vectors @ qn
        order = np.argsort(-sims)
        out = []
        for idx in order:
            meta = self.metadatas[idx]
            if where and not self._match(meta, where):
                continue
            out.append({
                "id": self.ids[idx],
                "score": float(sims[idx]),
                "metadata": meta,
                "document": self.documents[idx],
            })
            if len(out) >= top_k:
                break
        return out

    def count(self):
        return len(self.ids)

    def reset(self):
        self.ids, self.metadatas, self.documents, self.vectors = [], [], [], None
        if self.file.exists():
            self.file.unlink()


class VectorStore:
    """Facade over ChromaDB (preferred) or the numpy fallback."""

    def __init__(self):
        self.settings = get_settings()
        self.collection_name = self.settings.collection_name
        self._backend = None
        self._collection = None
        self._numpy: Optional[_NumpyStore] = None
        self._init()

    def _init(self):
        try:
            import chromadb

            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
            self._collection = client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
            self._client = client
            self._backend = "chromadb"
            logger.info("Vector store backend: chromadb (collection=%s)", self.collection_name)
        except Exception as exc:
            logger.warning("ChromaDB unavailable (%s). Using numpy vector store.", exc)
            self._numpy = _NumpyStore(self.settings.chroma_path, self.collection_name)
            self._backend = "numpy"

    @property
    def backend(self) -> str:
        return self._backend

    def add(self, ids: list[str], embeddings, metadatas: list[dict], documents: list[str]):
        # Chroma rejects None metadata values; coerce to primitives.
        clean_meta = [{k: ("" if v is None else v) for k, v in md.items()} for md in metadatas]
        emb_list = [list(map(float, e)) for e in embeddings]
        if self._backend == "chromadb":
            self._collection.upsert(
                ids=ids, embeddings=emb_list, metadatas=clean_meta, documents=documents
            )
        else:
            self._numpy.add(ids, emb_list, clean_meta, documents)

    def query(self, embedding, top_k: int = 10, where: Optional[dict] = None) -> list[dict]:
        if self._backend == "chromadb":
            kwargs: dict[str, Any] = {"query_embeddings": [list(map(float, embedding))], "n_results": top_k}
            if where:
                kwargs["where"] = _to_chroma_where(where)
            res = self._collection.query(**kwargs)
            out = []
            ids = res.get("ids", [[]])[0]
            dists = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            docs = res.get("documents", [[]])[0]
            for i, cid in enumerate(ids):
                dist = dists[i] if i < len(dists) else 1.0
                out.append({
                    "id": cid,
                    "score": 1.0 - float(dist),  # cosine distance -> similarity
                    "metadata": metas[i] if i < len(metas) else {},
                    "document": docs[i] if i < len(docs) else "",
                })
            return out
        return self._numpy.query(embedding, top_k=top_k, where=where)

    def count(self) -> int:
        if self._backend == "chromadb":
            try:
                return self._collection.count()
            except Exception:
                return 0
        return self._numpy.count()

    def reset(self):
        if self._backend == "chromadb":
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )
        else:
            self._numpy.reset()


def _to_chroma_where(where: dict) -> dict:
    """Translate a flat equality dict to Chroma's where syntax."""
    clauses = [{k: {"$eq": v}} for k, v in where.items()]
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def reset_vector_store():
    global _store
    _store = None
