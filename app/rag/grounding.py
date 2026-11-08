"""NLI-based factual-grounding checks with a lexical fallback.

Estimates whether each generated claim is entailed by its cited evidence. An NLI
result is treated as a signal, never as conclusive proof (spec 12.3). When the
NLI model is unavailable a conservative lexical entailment heuristic is used
(token coverage + numeric-consistency), so grounding checks always run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..core.config import get_settings
from ..core.logging import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9$%./-]+")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "at", "by", "with", "as", "that", "this", "it", "should", "review",
    "project", "based", "evidence", "available", "recommended", "action",
}


@dataclass
class GroundingResult:
    claim: str
    label: str  # entailment | contradiction | neutral
    score: float
    supported: bool
    method: str


class GroundingChecker:
    def __init__(self):
        self.settings = get_settings()
        self._model = None
        self._backend = None

    def _ensure(self):
        if self._backend is not None:
            return
        if not self.settings.nli_enabled:
            self._backend = "disabled"
            return
        try:
            from sentence_transformers import CrossEncoder

            logger.info("Loading NLI model '%s'...", self.settings.nli_model)
            self._model = CrossEncoder(self.settings.nli_model)
            self._backend = self.settings.nli_model
        except Exception as exc:
            logger.warning("NLI model unavailable (%s). Using lexical entailment.", exc)
            self._backend = "lexical-fallback"

    @property
    def backend(self) -> str:
        self._ensure()
        return self._backend

    def _content_tokens(self, text: str) -> set[str]:
        return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1}

    def _lexical(self, claim: str, evidence: str) -> GroundingResult:
        c_tokens = self._content_tokens(claim)
        e_tokens = self._content_tokens(evidence)
        if not c_tokens:
            return GroundingResult(claim, "neutral", 0.0, False, "lexical-fallback")
        coverage = len(c_tokens & e_tokens) / len(c_tokens)
        # Numbers in the claim must appear in the evidence.
        claim_nums = {n.replace(",", "") for n in _NUM_RE.findall(claim)}
        ev_nums = {n.replace(",", "") for n in _NUM_RE.findall(evidence)}
        nums_ok = claim_nums.issubset(ev_nums) if claim_nums else True
        supported = coverage >= self.settings.abstention_grounding_threshold and nums_ok
        label = "entailment" if supported else ("contradiction" if not nums_ok else "neutral")
        return GroundingResult(claim, label, round(coverage, 3), supported, "lexical-fallback")

    def check(self, claim: str, evidence: str) -> GroundingResult:
        self._ensure()
        if self._backend in ("disabled", "lexical-fallback"):
            return self._lexical(claim, evidence)
        try:
            # cross-encoder NLI returns logits for [contradiction, entailment, neutral]
            scores = self._model.predict([(evidence, claim)])
            import numpy as np

            arr = np.asarray(scores).reshape(-1)
            labels = ["contradiction", "entailment", "neutral"]
            if arr.shape[0] != 3:
                return self._lexical(claim, evidence)
            idx = int(arr.argmax())
            label = labels[idx]
            exp = np.exp(arr - arr.max())
            prob = float((exp / exp.sum())[idx])
            supported = label == "entailment" and prob >= self.settings.abstention_grounding_threshold
            return GroundingResult(claim, label, round(prob, 3), supported, self._backend)
        except Exception as exc:
            logger.warning("NLI prediction failed (%s); lexical fallback.", exc)
            return self._lexical(claim, evidence)


_checker: GroundingChecker | None = None


def get_grounding_checker() -> GroundingChecker:
    global _checker
    if _checker is None:
        _checker = GroundingChecker()
    return _checker


# Imperative verbs that mark a sentence as a *recommendation*, not a factual claim.
_IMPERATIVE_VERBS = {
    "review", "investigate", "conduct", "escalate", "submit", "initiate",
    "collect", "assess", "coordinate", "validate", "confirm", "re-baseline",
    "rebaseline", "consider", "ensure", "perform", "update", "check", "monitor",
    "prepare", "develop", "implement", "schedule", "request",
}


def is_factual_claim(sentence: str) -> bool:
    """True if the sentence asserts a fact (not an imperative recommendation)."""
    words = sentence.strip().split()
    if len(words) < 3:
        return False
    first = words[0].lower().strip(".,:")
    return first not in _IMPERATIVE_VERBS


def extract_claims(text: str, factual_only: bool = True) -> list[str]:
    """Split text into candidate claims; by default keep only factual (non-imperative) ones."""
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    out = []
    for s in sentences:
        s = s.strip()
        if len(s.split()) < 3:
            continue
        if factual_only and not is_factual_claim(s):
            continue
        out.append(s)
    return out
