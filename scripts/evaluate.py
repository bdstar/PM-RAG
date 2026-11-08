"""Evaluate the RAG pipeline against 3 representative queries.

Two queries have answers in the knowledge base; the third does NOT, and is used
to verify the anti-hallucination guardrail correctly refuses with "I don't know".

Run:  python scripts/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag_pipeline import get_pipeline  # noqa: E402
from app.schemas import AnswerStatus  # noqa: E402

# (question, expected_status, expected_keywords_if_answered)
TEST_CASES = [
    (
        "How many days per week can employees work remotely?",
        AnswerStatus.ANSWERED,
        ["3", "three"],
    ),
    (
        "How many days of paid time off (PTO) do employees get annually?",
        AnswerStatus.ANSWERED,
        ["20", "twenty"],
    ),
    (
        "How many free therapy sessions does the healthcare plan cover per year?",
        AnswerStatus.ANSWERED,
        ["12", "twelve"],
    ),
    (
        "What is AquaTech's policy on paying bonuses in cryptocurrency?",
        AnswerStatus.NOT_FOUND,  # not in any document -> guardrail must refuse
        [],
    ),
]


def main() -> int:
    pipeline = get_pipeline()
    if not pipeline.index_ready():
        print("Index not found - building it now...")
        pipeline.ingest(rebuild=True)

    print("=" * 78)
    print("AquaTech RAG - Evaluation (3 queries)")
    print("=" * 78)

    passed = 0
    for i, (question, expected_status, keywords) in enumerate(TEST_CASES, 1):
        resp = pipeline.query(question)
        status_ok = resp.status == expected_status
        kw_ok = (
            True
            if expected_status == AnswerStatus.NOT_FOUND
            else any(k.lower() in resp.answer.lower() for k in keywords)
        )
        ok = status_ok and kw_ok
        passed += int(ok)

        print(f"\n[{i}] Q: {question}")
        print(f"    expected: {expected_status.value:<10} got: {resp.status.value}")
        print(f"    grounded: {resp.grounded} | guardrail_triggered: {resp.guardrail_triggered}"
              f" | top_sim: {resp.sources[0].score if resp.sources else 'n/a'}")
        print(f"    A: {resp.answer}")
        print(f"    => {'PASS' if ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"RESULT: {passed}/{len(TEST_CASES)} test cases passed.")
    print("=" * 78)
    return 0 if passed == len(TEST_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
