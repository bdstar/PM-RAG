"""Benchmark construction: system-derived gold evidence + project-disjoint splits.

The "gold" relevant evidence is derived from the source records that generated a
scenario. This is clearly labelled as SYSTEM-DERIVED, not expert-validated
(spec 13: observed vs derived vs expert content must be distinguished). Experts
can later override via the scenario/annotation UI.
"""
from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.logging import get_logger
from ..database import models as m

logger = get_logger(__name__)

_SRC_KEY_TO_TYPE = {
    "schedule_history_ids": "schedule_history",
    "budget_history_ids": "budget_spend_history",
    "budget_schedule_ids": "citywide_budget_schedule",
}


def gold_chunk_keys(scenario: m.ExecutionScenario) -> list[str]:
    """Map a scenario's source record IDs to evidence chunk keys."""
    keys: list[str] = []
    src = scenario.source_record_ids or {}
    for src_key, dataset_type in _SRC_KEY_TO_TYPE.items():
        for rid in src.get(src_key, []) or []:
            keys.append(f"{dataset_type}:{rid}")
    return list(dict.fromkeys(keys))


def _split_for_project(project_id, ratios) -> str:
    h = int(hashlib.md5(str(project_id).encode()).hexdigest(), 16) % 1000 / 1000.0
    train, val = ratios[0], ratios[0] + ratios[1]
    if h < train:
        return "train"
    if h < val:
        return "val"
    return "test"


def build_benchmark(session: Session, max_scenarios: int | None = None,
                    ratios=(0.7, 0.15, 0.15), only_with_gold: bool = True) -> dict:
    scenarios = session.scalars(select(m.ExecutionScenario)).all()
    chunk_keys_present = set(session.scalars(select(m.EvidenceChunk.chunk_key)).all())

    included = 0
    gold_created = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    assignments: dict[str, str] = {}

    count = 0
    for sc in scenarios:
        gold = [k for k in gold_chunk_keys(sc) if k in chunk_keys_present]
        if only_with_gold and not gold:
            continue
        if max_scenarios and count >= max_scenarios:
            break
        count += 1

        # Materialise gold ScenarioEvidence (system-derived).
        existing = {
            r.chunk_id for r in session.scalars(
                select(m.ScenarioEvidence).where(m.ScenarioEvidence.scenario_id == sc.id)
            ).all()
        }
        chunks = session.scalars(
            select(m.EvidenceChunk).where(m.EvidenceChunk.chunk_key.in_(gold))
        ).all()
        for ch in chunks:
            if ch.id not in existing:
                session.add(m.ScenarioEvidence(
                    scenario_id=sc.id, chunk_id=ch.id, relevance=1,
                    is_distractor=False, note="system-derived (not expert-validated)",
                ))
                gold_created += 1

        split = _split_for_project(sc.project_id or sc.id, ratios)
        sc.split = split
        sc.include_in_benchmark = True
        split_counts[split] += 1
        assignments[sc.scenario_key] = split
        included += 1

    # Record the split definition.
    existing_split = session.scalar(select(m.BenchmarkSplit).where(m.BenchmarkSplit.name == "default"))
    if existing_split:
        existing_split.assignments = assignments
        existing_split.strategy = "project_disjoint_hash"
    else:
        session.add(m.BenchmarkSplit(
            name="default", strategy="project_disjoint_hash",
            description="Project-disjoint hashed split (train/val/test).",
            assignments=assignments,
        ))
    session.commit()
    logger.info("Benchmark built: %d scenarios, %d gold links, splits=%s",
                included, gold_created, split_counts)
    return {"scenarios_included": included, "gold_links_created": gold_created,
            "split_counts": split_counts}
