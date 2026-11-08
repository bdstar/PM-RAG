"""Configurable LLM backends.

Two backends are supported:

* ``huggingface`` - loads a real Hugging Face causal LM (default
  ``meta-llama/Llama-3.2-3B``) via LlamaIndex's ``HuggingFaceLLM``.
* ``stub`` - a dependency-free extractive fallback that does NOT require a GPU
  or the gated Llama weights. It returns the most relevant sentences from the
  retrieved context. Handy for development, CI, and low-resource machines.

The ``stub`` backend also activates automatically if the real model fails to
load (e.g. missing HF token / license, or out-of-memory), so the pipeline
always stays runnable.
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from .config import Settings

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...


class StubLLM(BaseLLM):
    """Extractive, no-dependency fallback.

    It looks for a ``CONTEXT:`` block and a ``QUESTION:`` line inside the prompt
    and returns the context sentences that best overlap with the question.
    Because it only ever echoes the supplied context, it cannot hallucinate.
    """

    name = "stub-extractive"

    _CTX_RE = re.compile(r"CONTEXT:\s*(.*?)\s*QUESTION:", re.DOTALL | re.IGNORECASE)
    _Q_RE = re.compile(r"QUESTION:\s*(.*?)(?:\n\n|$)", re.DOTALL | re.IGNORECASE)

    def generate(self, prompt: str) -> str:
        ctx_match = self._CTX_RE.search(prompt)
        q_match = self._Q_RE.search(prompt)
        context = ctx_match.group(1) if ctx_match else prompt
        question = q_match.group(1).strip() if q_match else ""

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        q_words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 2}

        scored = []
        for sent in sentences:
            s_words = set(re.findall(r"\w+", sent.lower()))
            overlap = len(q_words & s_words)
            if overlap:
                scored.append((overlap, sent))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return json.dumps(
                {
                    "status": "not_found",
                    "answer": "Information not provided.",
                    "confidence": 0.0,
                    "supporting_quotes": [],
                }
            )

        top = [re.sub(r"\s+", " ", s).strip() for _, s in scored[:3]]
        return json.dumps(
            {
                "status": "answered",
                "answer": " ".join(top),
                "confidence": 0.7,
                "supporting_quotes": [top[0]],
            }
        )


class HuggingFaceBackend(BaseLLM):
    """Real Hugging Face causal LM via LlamaIndex's HuggingFaceLLM wrapper."""

    def __init__(self, settings: Settings):
        from llama_index.llms.huggingface import HuggingFaceLLM

        self.name = settings.llm_model
        self._tokenizer = None

        model_kwargs: dict = {}
        if settings.load_in_4bit:
            try:
                import torch
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning("4-bit quantization unavailable (%s); loading full precision.", exc)

        if settings.hf_token:
            model_kwargs["token"] = settings.hf_token

        logger.info("Loading Hugging Face LLM '%s' (this can take a while)...", settings.llm_model)
        self._llm = HuggingFaceLLM(
            model_name=settings.llm_model,
            tokenizer_name=settings.llm_model,
            context_window=settings.llm_context_window,
            max_new_tokens=settings.max_new_tokens,
            generate_kwargs={
                "temperature": settings.llm_temperature,
                "do_sample": settings.llm_temperature > 0,
            },
            model_kwargs=model_kwargs,
            tokenizer_kwargs={"token": settings.hf_token} if settings.hf_token else {},
            device_map="auto",
        )
        # Cache tokenizer so we can apply its chat template for instruct models.
        self._tokenizer = getattr(self._llm, "_tokenizer", None)

    def _apply_chat_template(self, prompt: str) -> str:
        """Wrap the prompt with the model's chat template when available.

        Instruction-tuned models (Llama-3.2-*-Instruct, Qwen-Instruct, ...) follow
        instructions far more reliably when their chat template is applied. Base
        models without a chat template fall back to the raw prompt.
        """
        tok = self._tokenizer
        if tok is None or getattr(tok, "chat_template", None) is None:
            return prompt
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def generate(self, prompt: str) -> str:
        return self._llm.complete(self._apply_chat_template(prompt)).text


def build_llm(settings: Settings) -> BaseLLM:
    """Construct the configured LLM, falling back to the stub on any failure."""
    if settings.llm_backend == "stub":
        logger.info("Using stub (extractive) LLM backend.")
        return StubLLM()

    try:
        return HuggingFaceBackend(settings)
    except Exception as exc:
        logger.warning(
            "Could not load Hugging Face LLM '%s' (%s). Falling back to stub backend. "
            "For the real model, accept the license and set HF_TOKEN in .env.",
            settings.llm_model,
            exc,
        )
        return StubLLM()
