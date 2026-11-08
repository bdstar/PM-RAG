"""Prompt construction and the Pydantic-based anti-hallucination guardrail.

Grounding is enforced in three layers:

1. **Retrieval threshold** (in the pipeline) - if nothing relevant is retrieved
   we refuse before ever calling the LLM.
2. **Prompt contract** - the LLM is instructed to answer ONLY from the context
   and to emit JSON matching the ``GroundedAnswer`` schema.
3. **Pydantic validation** - the JSON is parsed and validated; any output that
   is unparseable, ungrounded, or lacks supporting quotes is converted into a
   safe "I don't know" refusal via ``GroundedAnswer.enforce()``.
"""
from __future__ import annotations

import json
import re

from .schemas import REFUSAL_TEXT, AnswerStatus, GroundedAnswer

SYSTEM_INSTRUCTIONS = """You are AquaTech Corporation's policy assistant.
Answer the employee's QUESTION using ONLY the information in CONTEXT below.

Strict rules:
- If the answer is not fully contained in the CONTEXT, you MUST say it is not found.
- Never use outside knowledge. Never guess. Never invent numbers or policies.
- Every claim in your answer must be supported by a verbatim quote from CONTEXT.

Respond with ONLY a single JSON object, no extra text, in exactly this format:
{
  "status": "answered" or "not_found",
  "answer": "<concise answer, or 'I don't know' if not found>",
  "confidence": <number between 0 and 1>,
  "supporting_quotes": ["<verbatim snippet from context>", ...]
}
"""

PROMPT_TEMPLATE = """{system}

CONTEXT:
{context}

QUESTION: {question}

JSON:"""


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(
        system=SYSTEM_INSTRUCTIONS, context=context.strip(), question=question.strip()
    )


def _strip_fences(raw: str) -> str:
    """Remove ```json ... ``` markdown code fences."""
    return re.sub(r"```(?:json)?|```", "", raw or "").strip()


def _extract_json(raw: str) -> dict | None:
    """Best-effort extraction of the first JSON object from model output."""
    text = _strip_fences(raw)

    # Direct parse first.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Find the first balanced {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)  # trailing commas
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    # Last resort: pull individual fields out with regex. This survives
    # malformed arrays / missing commas that small models sometimes emit.
    status_m = re.search(r'"status"\s*:\s*"(answered|not_found)"', text, re.IGNORECASE)
    answer_m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if status_m or answer_m:
        conf_m = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        return {
            "status": (status_m.group(1).lower() if status_m else "answered"),
            "answer": (json.loads(f'"{answer_m.group(1)}"') if answer_m else ""),
            "confidence": float(conf_m.group(1)) if conf_m else 0.5,
            "supporting_quotes": [],
        }
    return None


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "at", "by", "with", "as", "that", "this", "it", "do", "does", "up",
    "per", "you", "your", "our", "we", "they", "from", "can", "will", "may",
    "i", "dont", "know", "not", "provided", "information",
}


def _content_tokens(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9$%]+", (text or "").lower())
        if w not in _STOPWORDS and len(w) > 1
    }


def verify_faithful(answer: str, context: str, min_overlap: float = 0.5) -> bool:
    """Lightweight faithfulness guardrail.

    Returns True if the content words of ``answer`` are sufficiently covered by
    ``context``. This catches hallucinations where the model claims the answer is
    grounded but introduces facts (numbers, names) absent from the context.
    Numeric tokens must always appear in the context.
    """
    ans_tokens = _content_tokens(answer)
    if not ans_tokens:
        return False
    ctx_tokens = _content_tokens(context)

    # Any number cited in the answer must be present in the context.
    ans_numbers = {t for t in ans_tokens if any(c.isdigit() for c in t)}
    if ans_numbers and not ans_numbers.issubset(ctx_tokens):
        return False

    covered = len(ans_tokens & ctx_tokens) / len(ans_tokens)
    return covered >= min_overlap


def parse_grounded_answer(raw_output: str) -> GroundedAnswer:
    """Parse raw LLM text into a validated, guardrailed ``GroundedAnswer``."""
    data = _extract_json(raw_output)

    if data is None:
        # Could not get structured output -> refuse rather than risk hallucinating.
        return GroundedAnswer(
            status=AnswerStatus.NOT_FOUND,
            answer=REFUSAL_TEXT,
            confidence=0.0,
            supporting_quotes=[],
        )

    try:
        answer = GroundedAnswer.model_validate(data)
    except Exception:
        return GroundedAnswer(
            status=AnswerStatus.NOT_FOUND,
            answer=REFUSAL_TEXT,
            confidence=0.0,
            supporting_quotes=[],
        )

    return answer.enforce()
