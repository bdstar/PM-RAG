"""Configurable LLM/SLM provider adapters.

Supported backends (via ``LLM_BACKEND``):
  * ``huggingface`` - local Hugging Face causal LM (LlamaIndex wrapper).
  * ``openai``      - any OpenAI-compatible chat endpoint (needs API key).
  * ``ollama``      - local Ollama server.
  * ``stub``        - no model; signals the generator to use its evidence-only
                      rule-based generator (which cannot hallucinate).

Any backend that fails to initialise automatically falls back to the stub so the
platform always runs. ``is_stub`` tells the generator to use the grounded
template generator instead of a free-form model.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.config import Settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class BaseLLM(ABC):
    name: str = "base"
    is_stub: bool = False

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class StubLLM(BaseLLM):
    name = "stub-evidence-only"
    is_stub = True

    def generate(self, prompt: str) -> str:  # pragma: no cover - never called directly
        return ""


class HuggingFaceLLM(BaseLLM):
    def __init__(self, settings: Settings):
        from llama_index.llms.huggingface import HuggingFaceLLM as LIHF

        self.name = settings.llm_model
        model_kwargs: dict = {}
        if settings.load_in_4bit:
            try:
                import torch
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("4-bit unavailable (%s); full precision.", exc)
        if settings.hf_token:
            model_kwargs["token"] = settings.hf_token
        self._llm = LIHF(
            model_name=settings.llm_model,
            tokenizer_name=settings.llm_model,
            context_window=settings.llm_context_window,
            max_new_tokens=settings.max_new_tokens,
            generate_kwargs={"temperature": settings.llm_temperature,
                             "do_sample": settings.llm_temperature > 0},
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"token": settings.hf_token} if settings.hf_token else {},
            device_map="auto",
        )
        self._tok = getattr(self._llm, "_tokenizer", None)

    def _chat(self, prompt: str) -> str:
        tok = self._tok
        if tok is not None and getattr(tok, "chat_template", None):
            try:
                return tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                return prompt
        return prompt

    def generate(self, prompt: str) -> str:
        return self._llm.complete(self._chat(prompt)).text


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        import httpx

        self.name = settings.llm_model
        self._base = settings.openai_base_url.rstrip("/")
        self._key = settings.openai_api_key
        self._model = settings.llm_model
        self._temp = settings.llm_temperature
        self._max = settings.max_new_tokens
        self._client = httpx.Client(timeout=120)

    def generate(self, prompt: str) -> str:
        resp = self._client.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self._temp,
                "max_tokens": self._max,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


class OllamaLLM(BaseLLM):
    def __init__(self, settings: Settings):
        import httpx

        self.name = settings.llm_model
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.llm_model
        self._temp = settings.llm_temperature
        self._client = httpx.Client(timeout=300)
        # Fail fast if server not reachable.
        self._client.get(f"{self._base}/api/tags")

    def generate(self, prompt: str) -> str:
        resp = self._client.post(
            f"{self._base}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False,
                  "options": {"temperature": self._temp}},
        )
        resp.raise_for_status()
        return resp.json().get("response", "")


def build_llm(settings: Settings) -> BaseLLM:
    backend = (settings.llm_backend or "stub").lower()
    if backend == "stub":
        logger.info("LLM backend: stub (evidence-only generator).")
        return StubLLM()
    try:
        if backend == "huggingface":
            return HuggingFaceLLM(settings)
        if backend == "openai":
            return OpenAICompatibleLLM(settings)
        if backend == "ollama":
            return OllamaLLM(settings)
        raise ValueError(f"Unknown LLM_BACKEND '{backend}'")
    except Exception as exc:
        logger.warning(
            "LLM backend '%s' unavailable (%s). Falling back to evidence-only "
            "stub generator.", backend, exc,
        )
        return StubLLM()


_llm: BaseLLM | None = None


def get_llm(settings: Settings) -> BaseLLM:
    global _llm
    if _llm is None:
        _llm = build_llm(settings)
    return _llm


def reset_llm():
    global _llm
    _llm = None
