"""Central configuration for the PM-RAG platform.

All settings are loaded from environment variables / a local ``.env`` file via
``pydantic-settings``. Credentials are never hard-coded: database and model
credentials are read from the environment and default to safe, local values so
the platform runs out-of-the-box without any paid external service.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the "app" package directory.
ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings; override any value via ``.env`` or the environment."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Application ----
    app_name: str = "PM-RAG"
    app_env: str = "development"
    log_level: str = "INFO"

    # ---- Paths ----
    dataset_dir: str = str(ROOT_DIR.parent / "dataset")
    storage_dir: str = "storage"
    knowledge_dir: str = "knowledge"
    reports_dir: str = "reports"

    # ---- Database ----
    db_backend: str = "sqlite"  # sqlite | mysql | postgresql
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "pm_rag"
    db_user: str = "root"
    db_password: str = ""
    database_url: Optional[str] = None  # explicit override

    # ---- Vector store ----
    chroma_dir: str = "storage/chroma"
    collection_name: str = "pm_rag_evidence"

    # ---- Embeddings ----
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_fallback_dim: int = 384

    # ---- Chunking ----
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- Retrieval ----
    top_k: int = 8
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    rrf_k: int = 60
    similarity_threshold: float = 0.15

    # ---- Reranking ----
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True

    # ---- LLM ----
    llm_backend: str = "huggingface"  # huggingface | openai | ollama | stub
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_new_tokens: int = 512
    llm_temperature: float = 0.1
    llm_context_window: int = 4096
    load_in_4bit: bool = False
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    hf_token: Optional[str] = None

    # ---- NLI grounding ----
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_enabled: bool = True

    # ---- Abstention ----
    min_evidence_for_answer: int = 1
    # Per-claim NLI entailment probability needed to call a claim "supported".
    abstention_grounding_threshold: float = 0.5
    # Minimum fraction of factual claims that must be supported to avoid
    # grounding-based abstention. Default (0.01) abstains only when NO factual
    # claim is supported; raise it (e.g. 0.5) for the stricter H4 ablation.
    abstention_min_support_rate: float = 0.01

    # ------------------------------------------------------------------
    # Derived absolute paths
    # ------------------------------------------------------------------
    def _abs(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (ROOT_DIR / p)

    @property
    def dataset_path(self) -> Path:
        return self._abs(self.dataset_dir)

    @property
    def storage_path(self) -> Path:
        return self._abs(self.storage_dir)

    @property
    def knowledge_path(self) -> Path:
        return self._abs(self.knowledge_dir)

    @property
    def reports_path(self) -> Path:
        return self._abs(self.reports_dir)

    @property
    def chroma_path(self) -> Path:
        return self._abs(self.chroma_dir)

    @property
    def sqlalchemy_url(self) -> str:
        """Build the SQLAlchemy connection URL from config.

        An explicit ``DATABASE_URL`` always wins. Otherwise the URL is derived
        from ``DB_BACKEND`` and the individual credential fields (never hard
        coded). SQLite is the default so the platform runs with no DB server.
        """
        if self.database_url:
            return self.database_url

        backend = (self.db_backend or "sqlite").lower()
        if backend == "sqlite":
            self.storage_path.mkdir(parents=True, exist_ok=True)
            db_file = self.storage_path / "pm_rag.db"
            return f"sqlite:///{db_file.as_posix()}"
        if backend == "mysql":
            pwd = self.db_password or ""
            return (
                f"mysql+pymysql://{self.db_user}:{pwd}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}?charset=utf8mb4"
            )
        if backend == "postgresql":
            pwd = self.db_password or ""
            return (
                f"postgresql+psycopg2://{self.db_user}:{pwd}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}"
            )
        raise ValueError(f"Unsupported DB_BACKEND '{self.db_backend}'")

    def ensure_dirs(self) -> None:
        for p in (self.storage_path, self.knowledge_path, self.reports_path, self.chroma_path):
            p.mkdir(parents=True, exist_ok=True)
        (self.storage_path / "uploads").mkdir(parents=True, exist_ok=True)
        (self.storage_path / "logs").mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
