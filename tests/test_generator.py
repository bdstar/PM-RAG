"""Unit tests for the deterministic, evidence-only generator."""
from __future__ import annotations

from app.rag import generator as GEN


def _evidence():
    return [
        {
            "chunk_key": "EV-1",
            "text": (
                "Record Type: budget_snapshot\n"
                "Reporting Period: 202305\n"
                "Budget Variance: 5000\n"
                "Managing Agency: DEP"
            ),
        },
        {"chunk_key": "EV-2", "text": "Record Type: schedule\nVariance (Day): 45"},
    ]


def test_rule_based_abstains_without_evidence():
    out = GEN.rule_based_generate("budget_variance", "problem", "ctx", [])
    assert out["evidence_sufficient"] is False
    assert out["recommended_actions"] == []
    assert out["abstention_reason"]


def test_rule_based_grounds_in_evidence():
    out = GEN.rule_based_generate("budget_variance", "Budget change detected.", "ctx", _evidence())
    assert out["evidence_sufficient"] is True
    assert len(out["recommended_actions"]) == 1
    action = out["recommended_actions"][0]
    # Cites only real evidence keys.
    assert set(action["supporting_evidence_ids"]).issubset({"EV-1", "EV-2"})
    # Budget category must require human approval (spec 11/19).
    assert action["requires_human_approval"] is True
    # Evidence summary references the actual cited record / facts.
    assert "EV-1" in out["evidence_summary"]


def test_extract_json_tolerates_fences_and_trailing_commas():
    raw = '```json\n{"a": 1, "recommended_actions": [],}\n```'
    parsed = GEN._extract_json(raw)
    assert parsed["a"] == 1


def test_key_facts_skips_na_and_meta():
    facts = GEN._key_facts(
        "Record Type: x\nProject Key: PID:1\nBudget Variance: 5000\nBorough: N/A"
    )
    assert "budget variance is 5000" in facts
    assert "record type" not in facts
    assert "n/a" not in facts.lower()
