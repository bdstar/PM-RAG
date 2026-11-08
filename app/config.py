"""Central configuration loaded from environment variables / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = parent of the "app" package directory.
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings. Values can be overridden via a .env file."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- Paths ----
    knowledge_dir: str = "knowledge"
    storage_dir: str = "storage"
    collection_name: str = "aquatech_policies"

    # ---- Embeddings ----
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ---- Chunking ----
    chunk_size: int = 512
    chunk_overlap: int = 64

    # ---- Retrieval ----
    top_k: int = 4
    similarity_threshold: float = 0.30

    # ---- LLM ----
    llm_backend: str = "huggingface"  # "huggingface" | "stub"
    llm_model: str = "meta-llama/Llama-3.2-3B"
    max_new_tokens: int = 320
    llm_temperature: float = 0.1
    llm_context_window: int = 4096
    load_in_4bit: bool = False
    hf_token: str | None = None

    # ---- Derived absolute paths ----
    @property
    def knowledge_path(self) -> Path:
        return ROOT_DIR / self.knowledge_dir

    @property
    def storage_path(self) -> Path:
        return ROOT_DIR / self.storage_dir

    @property
    def chroma_path(self) -> Path:
        return self.storage_path / "chroma"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
