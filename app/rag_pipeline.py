"""End-to-end RAG pipeline built on LlamaIndex + ChromaDB.

Stages:
  1. Load PDF/TXT documents from the knowledge folder.
  2. Split them into overlapping semantic chunks (SentenceSplitter).
  3. Embed chunks with a Hugging Face sentence-transformers model.
  4. Persist embeddings in a local ChromaDB collection.
  5. Retrieve the top-k chunks for a query (with a similarity guardrail).
  6. Generate a grounded, Pydantic-validated answer with the configured LLM.
"""
from __future__ import annotations

import logging
import time

import chromadb
from llama_index.core import Settings as LISettings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import Settings, get_settings
from .guardrails import build_prompt, parse_grounded_answer, verify_faithful
from .llm import BaseLLM, build_llm
from .schemas import (
    AnswerStatus,
    GroundedAnswer,
    QueryResponse,
    REFUSAL_TEXT,
    SourceChunk,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Holds the embedding model, vector store, index and LLM."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._embed_model: HuggingFaceEmbedding | None = None
        self._chroma_client: chromadb.api.ClientAPI | None = None
        self._index: VectorStoreIndex | None = None
        self._llm: BaseLLM | None = None
        self._splitter = SentenceSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Lazy components
    # ------------------------------------------------------------------
    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        if self._embed_model is None:
            logger.info("Loading embedding model '%s'...", self.settings.embed_model)
            self._embed_model = HuggingFaceEmbedding(model_name=self.settings.embed_model)
            LISettings.embed_model = self._embed_model
            LISettings.llm = None  # we drive the LLM ourselves, not via LlamaIndex query engine
        return self._embed_model

    @property
    def chroma_client(self) -> "chromadb.api.ClientAPI":
        if self._chroma_client is None:
            self.settings.chroma_path.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        return self._chroma_client

    def _get_collection(self):
        return self.chroma_client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = build_llm(self.settings)
        return self._llm

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _load_documents(self) -> list[Document]:
        from llama_index.core import SimpleDirectoryReader

        kpath = self.settings.knowledge_path
        if not kpath.exists() or not any(kpath.iterdir()):
            raise FileNotFoundError(
                f"No documents found in '{kpath}'. Add a .pdf/.txt file or run "
                f"`python scripts/generate_knowledge_pdf.py`."
            )
        reader = SimpleDirectoryReader(
            input_dir=str(kpath),
            required_exts=[".pdf", ".txt", ".md"],
            recursive=True,
        )
        docs = reader.load_data()
        logger.info("Loaded %d document section(s) from %s", len(docs), kpath)
        return docs

    def chunk_count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def index_ready(self) -> bool:
        return self.chunk_count() > 0

    def ingest(self, rebuild: bool = True) -> dict:
        """Load -> chunk -> embed -> store. Returns ingestion stats."""
        if rebuild:
            try:
                self.chroma_client.delete_collection(self.settings.collection_name)
                logger.info("Cleared existing collection '%s'.", self.settings.collection_name)
            except Exception:
                pass
            self._index = None

        documents = self._load_documents()
        nodes = self._splitter.get_nodes_from_documents(documents)
        logger.info(
            "Split into %d chunks (size=%d, overlap=%d).",
            len(nodes),
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

        collection = self._get_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=False,
        )

        return {
            "documents_loaded": len(documents),
            "chunks_indexed": len(nodes),
            "collection": self.settings.collection_name,
            "embed_model": self.settings.embed_model,
        }

    def _load_existing_index(self) -> VectorStoreIndex:
        collection = self._get_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=self.embed_model
        )

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            if self.index_ready():
                self._index = self._load_existing_index()
            else:
                logger.info("No existing index found; building one now.")
                self.ingest(rebuild=True)
        return self._index

    # ------------------------------------------------------------------
    # Retrieval + generation
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None):
        k = top_k or self.settings.top_k
        retriever = self.index.as_retriever(similarity_top_k=k)
        return retriever.retrieve(query)

    def query(self, query: str, top_k: int | None = None) -> QueryResponse:
        start = time.perf_counter()
        nodes = self.retrieve(query, top_k)

        sources = [
            SourceChunk(
                rank=i + 1,
                score=round(float(n.score or 0.0), 4),
                text=n.node.get_content().strip(),
                source=str(n.node.metadata.get("file_name", "knowledge")),
            )
            for i, n in enumerate(nodes)
        ]

        top_score = max((s.score for s in sources), default=0.0)

        # Guardrail #1: nothing relevant retrieved -> refuse before calling the LLM.
        if not sources or top_score < self.settings.similarity_threshold:
            return QueryResponse(
                query=query,
                status=AnswerStatus.NOT_FOUND,
                answer=REFUSAL_TEXT,
                confidence=0.0,
                grounded=False,
                sources=sources,
                guardrail_triggered=True,
                backend=self.llm.name,
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

        # Build grounded prompt from retrieved context.
        context = "\n\n---\n\n".join(
            f"[Source {s.rank}] {s.text}" for s in sources
        )
        prompt = build_prompt(query, context)

        raw = self.llm.generate(prompt)

        # Guardrail #2: validate/normalise structured output with Pydantic and
        # honour the model's own grounding judgment.
        grounded: GroundedAnswer = parse_grounded_answer(raw)

        # Guardrail #3: context-faithfulness check. Even if the model claims the
        # answer is grounded, refuse if it is not actually supported by the
        # retrieved context (catches hallucinated facts / numbers).
        if grounded.status == AnswerStatus.ANSWERED and not verify_faithful(
            grounded.answer, context
        ):
            grounded = grounded.as_refusal()

        guardrail_triggered = grounded.status == AnswerStatus.NOT_FOUND

        return QueryResponse(
            query=query,
            status=grounded.status,
            answer=grounded.answer,
            confidence=round(grounded.confidence, 3),
            grounded=grounded.status == AnswerStatus.ANSWERED,
            sources=sources,
            guardrail_triggered=guardrail_triggered,
            backend=self.llm.name,
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
