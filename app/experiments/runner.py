"""Experiment runner.

Executes a baseline/PM-RAG configuration over a benchmark split, generating a
recommendation per scenario, enforcing temporal validity, computing metrics
(retrieval / attribution / factual consistency / abstention / efficiency) and
persisting everything for reproducibility (spec 13, 14, 18).
"""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.logging import get_logger
from ..database import models as m
from ..rag.embeddings import get_embedder
from ..rag.llm_provider import get_llm
from ..recommendations.schemas import GenerateRequest
from ..recommendations.service import generate_recommendation
from . import metrics
from .baselines import get_baseline
from .benchmark import gold_chunk_keys

logger = get_logger(__name__)

K_VALUES = [1, 3, 5, 8]


def _reproducibility(session: Session, config: dict, seed: int) -> dict:
    settings = get_settings()
    files = session.scalars(select(m.DatasetFile)).all()
    index_version = session.scalar(
        select(m.RetrievalIndexVersion).order_by(m.RetrievalIndexVersion.id.desc())
    )
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "embed_model": settings.embed_model,
        "embed_backend": get_embedder().backend,
        "reranker_model": settings.reranker_model,
        "generator_model": settings.llm_model,
        "llm_backend": settings.llm_backend,
        "prompt_version": "pmrag-gen-v1",
        "index_version": index_version.version_key if index_version else None,
        "dataset_checksums": {f.file_name: f.checksum_sha256 for f in files},
        "config": config,
        "random_seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_experiment_run(
    session: Session, *, experiment_id: int, baseline: str = "pm_rag_full",
    config_overrides: Optional[dict] = None, split: Optional[str] = "test",
    max_scenarios: Optional[int] = None, seed: int = 42,
    scenario_ids: Optional[list[int]] = None,
) -> dict:
    settings = get_settings()
    base_cfg = get_baseline(baseline)
    if config_overrides:
        base_cfg.update(config_overrides)

    top_k = base_cfg.get("top_k", settings.top_k)

    # Persist configuration + run.
    cfg_row = m.ExperimentConfiguration(
        experiment_id=experiment_id, config=base_cfg, benchmark_split=split,
        retrieval_strategy=base_cfg.get("retrieval_mode"), top_k=top_k,
        reranker_enabled=base_cfg.get("reranker_enabled", True),
        llm_model=settings.llm_model, prompt_version="pmrag-gen-v1",
        grounding_enabled=base_cfg.get("grounding_enabled", True),
        abstention_enabled=base_cfg.get("abstention_enabled", True),
        citation_enabled=base_cfg.get("citation_enabled", True),
    )
    session.add(cfg_row)
    session.flush()

    run = m.ExperimentRun(
        experiment_id=experiment_id, configuration_id=cfg_row.id, baseline=baseline,
        status="running", started_at=datetime.now(timezone.utc), random_seed=seed,
        reproducibility=_reproducibility(session, base_cfg, seed),
    )
    session.add(run)
    session.commit()

    # Select benchmark scenarios. An explicit scenario_ids list (used to evaluate
    # the SAME stratified set across every configuration) takes precedence and
    # preserves the given order; otherwise fall back to the split + cap.
    if scenario_ids:
        found = {s.id: s for s in session.scalars(
            select(m.ExecutionScenario).where(m.ExecutionScenario.id.in_(scenario_ids))
        ).all()}
        scenarios = [found[i] for i in scenario_ids if i in found]
    else:
        q = select(m.ExecutionScenario).where(m.ExecutionScenario.include_in_benchmark.is_(True))
        if split:
            q = q.where(m.ExecutionScenario.split == split)
        scenarios = session.scalars(q).all()
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]

    chunk_keys_present = set(session.scalars(select(m.EvidenceChunk.chunk_key)).all())

    per_scenario = {f"recall@{k}": [] for k in K_VALUES}
    per_scenario.update({f"precision@{k}": [] for k in K_VALUES})
    per_scenario.update({f"ndcg@{k}": [] for k in K_VALUES})
    per_scenario["mrr"] = []
    attr_p, attr_r, attr_f1 = [], [], []
    grounding_rows_all: list[dict] = []
    abstention_records: list[dict] = []
    latencies: list[dict] = []

    try:
        for sc in scenarios:
            req = GenerateRequest(
                scenario_id=sc.id,
                retrieval_mode=base_cfg.get("retrieval_mode", "hybrid_temporal"),
                top_k=top_k,
                reranker_enabled=base_cfg.get("reranker_enabled", True),
                grounding_enabled=base_cfg.get("grounding_enabled", True),
                citation_enabled=base_cfg.get("citation_enabled", True),
                abstention_enabled=base_cfg.get("abstention_enabled", True),
                enabled_retrievers=base_cfg.get("enabled_retrievers", ["bm25", "dense"]),
                force_rule_based=base_cfg.get("force_rule_based", False),
            )
            out = generate_recommendation(session, req, experiment_run_id=run.id)

            retrieved_ids = [e["chunk_key"] for e in out["retrieved_evidence"]]
            gold = {k for k in gold_chunk_keys(sc) if k in chunk_keys_present}

            sc_metrics = {}
            for k in K_VALUES:
                r = metrics.recall_at_k(retrieved_ids, gold, k)
                p = metrics.precision_at_k(retrieved_ids, gold, k)
                n = metrics.ndcg_at_k(retrieved_ids, gold, k)
                if r is not None:
                    per_scenario[f"recall@{k}"].append(r); sc_metrics[f"recall@{k}"] = r
                if p is not None:
                    per_scenario[f"precision@{k}"].append(p); sc_metrics[f"precision@{k}"] = p
                if n is not None:
                    per_scenario[f"ndcg@{k}"].append(n); sc_metrics[f"ndcg@{k}"] = n
            mrr_v = metrics.mrr(retrieved_ids, gold)
            if mrr_v is not None:
                per_scenario["mrr"].append(mrr_v); sc_metrics["mrr"] = mrr_v

            cited = []
            for a in out["recommended_actions"]:
                cited.extend(a.get("supporting_evidence_ids", []))
            valid_ids = {d["cited_id"] for d in out["validation"]["citations"]["details"] if d["valid"]}
            attr = metrics.attribution(cited, valid_ids, gold)
            if attr["precision"] is not None:
                attr_p.append(attr["precision"])
            if attr["recall"] is not None:
                attr_r.append(attr["recall"])
            if attr["f1"] is not None:
                attr_f1.append(attr["f1"])

            grounding_rows_all.extend(out["validation"]["grounding"]["results"])

            should_abstain = None
            if sc.evidence_sufficient_label is not None:
                should_abstain = not sc.evidence_sufficient_label
            abstention_records.append({"abstained": out["abstained"], "should_abstain": should_abstain})

            latencies.append(out["latency"])

            session.add(m.ExperimentPrediction(
                run_id=run.id, scenario_id=sc.id, result_id=out["result_id"],
                retrieved_chunk_ids={"ids": retrieved_ids}, cited_chunk_ids={"ids": cited},
                abstained=out["abstained"], metrics=sc_metrics,
            ))
            session.commit()

        # ---- Aggregate + persist ----
        _persist_retrieval_metrics(session, run.id, per_scenario)
        _persist_recommendation_metrics(
            session, run.id, attr_p, attr_r, attr_f1, grounding_rows_all, abstention_records
        )
        _persist_performance_metrics(session, run.id, latencies)

        run.status = "completed"
        run.num_scenarios = len(scenarios)
        run.finished_at = datetime.now(timezone.utc)
        session.commit()
    except Exception as exc:  # pragma: no cover
        session.rollback()
        run = session.get(m.ExperimentRun, run.id)
        run.status = "failed"
        run.message = str(exc)[:2000]
        run.finished_at = datetime.now(timezone.utc)
        session.commit()
        logger.exception("Experiment run failed")
        raise

    return {"run_id": run.id, "baseline": baseline, "num_scenarios": len(scenarios),
            "status": run.status}


def _persist_retrieval_metrics(session, run_id, per_scenario):
    for name, values in per_scenario.items():
        if not values:
            continue
        mean_v, lo, hi = metrics.bootstrap_ci(values)
        k = None
        if "@" in name:
            k = int(name.split("@")[1])
        session.add(m.RetrievalMetric(
            run_id=run_id, metric_name=name.split("@")[0] if "@" in name else name,
            k=k, value=mean_v, ci_low=lo, ci_high=hi,
        ))
    session.commit()


def _persist_recommendation_metrics(session, run_id, attr_p, attr_r, attr_f1,
                                    grounding_rows, abstention_records):
    def _add(name, values, note=None):
        mean_v, lo, hi = metrics.bootstrap_ci(values)
        if mean_v is None:
            return
        session.add(m.RecommendationMetric(
            run_id=run_id, metric_name=name, value=mean_v, ci_low=lo, ci_high=hi,
            sample_size=len(values), note=note,
        ))

    _add("attribution_precision", attr_p)
    _add("attribution_recall", attr_r)
    _add("attribution_f1", attr_f1)

    fc = metrics.factual_consistency(grounding_rows)
    for name in ("supported_rate", "unsupported_rate", "contradiction_rate"):
        if fc.get(name) is not None:
            session.add(m.RecommendationMetric(
                run_id=run_id, metric_name=name, value=fc[name],
                sample_size=fc["n_claims"], note="over all factual claims",
            ))

    ab = metrics.abstention_metrics(abstention_records)
    for name in ("precision", "recall", "evidence_insufficient_detection_rate",
                 "inappropriate_recommendation_rate"):
        if ab.get(name) is not None:
            session.add(m.RecommendationMetric(
                run_id=run_id, metric_name=f"abstention_{name}", value=ab[name],
                sample_size=ab["n"],
            ))
    # Abstention rate (always computable).
    if abstention_records:
        rate = sum(1 for r in abstention_records if r["abstained"]) / len(abstention_records)
        session.add(m.RecommendationMetric(
            run_id=run_id, metric_name="abstention_rate", value=rate,
            sample_size=len(abstention_records),
        ))
    session.commit()


def _persist_performance_metrics(session, run_id, latencies):
    eff = metrics.efficiency(latencies)
    unit = {"throughput_per_min": "rec/min"}
    for name, value in eff.items():
        if value is None:
            continue
        session.add(m.PerformanceMetric(
            run_id=run_id, metric_name=name, value=value,
            unit=unit.get(name, "ms"),
        ))
    session.commit()
