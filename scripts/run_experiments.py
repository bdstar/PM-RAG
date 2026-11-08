"""Run a research-paper experiment campaign.

Builds a project-disjoint benchmark (system-derived gold evidence), then
evaluates a fixed N-scenario *test* set across the full baseline suite plus key
ablations. All configurations evaluate the SAME scenarios for a fair comparison.
Every metric is computed from actual runs and persisted with bootstrap CIs; a
summary table and CSV export are written to the reports directory.

Usage (PowerShell):
    python scripts/run_experiments.py                 # 100 test scenarios, full matrix
    python scripts/run_experiments.py --eval-n 50     # smaller/faster
    python scripts/run_experiments.py --no-ablations  # baselines only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import random  # noqa: E402

from sqlalchemy import func, select  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.database import init_db, session_scope  # noqa: E402
from app.database import models as m  # noqa: E402
from app.experiments.benchmark import build_benchmark  # noqa: E402
from app.experiments.reporting import compare_runs, export_runs_csv, run_summary  # noqa: E402
from app.experiments.runner import run_experiment_run  # noqa: E402

setup_logging()
logger = get_logger("experiments")

# Core baseline comparison (RQ1/RQ2/H1/H3).
BASELINE_SUITE = [
    "no_retrieval_llm",
    "keyword_template",
    "bm25_only",
    "dense_only",
    "hybrid_no_rerank",
    "pm_rag_full",
    "nearest_historical_case",
    "rule_based_alerts",
]

# Ablations on top of PM-RAG full (RQ3/RQ4/H2/H4): (label, overrides).
ABLATIONS = [
    ("pm_rag_no_grounding", {"grounding_enabled": False}),
    ("pm_rag_no_abstention", {"abstention_enabled": False}),
    ("pm_rag_no_citation", {"citation_enabled": False}),
]


def ensure_benchmark(eval_n: int, build_cap, rebuild: bool) -> dict:
    with session_scope() as s:
        test_by_type = dict(s.execute(
            select(m.ExecutionScenario.problem_type, func.count())
            .where(m.ExecutionScenario.include_in_benchmark.is_(True),
                   m.ExecutionScenario.split == "test")
            .group_by(m.ExecutionScenario.problem_type)
        ).all())
        test_n = sum(test_by_type.values())
        # Rebuild if asked, if too few test scenarios, or if coverage is thin
        # (we want the 4 actionable problem types represented).
        actionable = {"schedule_delay", "forecast_deterioration",
                      "budget_variance", "repeated_forecast_revision"}
        covered = actionable & set(test_by_type)
        if not rebuild and test_n >= eval_n and len(covered) >= 4:
            logger.info("Benchmark reused: %d test scenarios, types=%s", test_n, test_by_type)
            return {"reused": True, "test_scenarios": test_n}
        logger.info("Building benchmark (cap=%s, gold-only)...", build_cap)
        res = build_benchmark(s, max_scenarios=build_cap, only_with_gold=True)
        logger.info("Benchmark: %s", res)
        return res


def select_eval_scenarios(eval_n: int, seed: int) -> list[int]:
    """Deterministic, stratified test-set sample balanced across problem types."""
    rng = random.Random(seed)
    with session_scope() as s:
        by_type: dict[str, list[int]] = {}
        rows = s.execute(
            select(m.ExecutionScenario.problem_type, m.ExecutionScenario.id)
            .where(m.ExecutionScenario.include_in_benchmark.is_(True),
                   m.ExecutionScenario.split == "test")
        ).all()
        for ptype, sid in rows:
            by_type.setdefault(ptype, []).append(sid)
    types = sorted(by_type)
    per_type = max(1, eval_n // len(types))
    picked: list[int] = []
    for t in types:
        ids = by_type[t][:]
        rng.shuffle(ids)
        picked.extend(ids[:per_type])
    # Top up to exactly eval_n from the remaining pool if rounding left a gap.
    if len(picked) < eval_n:
        remaining = [sid for t in types for sid in by_type[t] if sid not in set(picked)]
        rng.shuffle(remaining)
        picked.extend(remaining[: eval_n - len(picked)])
    picked = picked[:eval_n]
    rng.shuffle(picked)
    logger.info("Selected %d eval scenarios across %d types (%s per type).",
                len(picked), len(types), per_type)
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-n", type=int, default=100, help="Test scenarios per configuration.")
    ap.add_argument("--build-cap", type=int, default=None,
                    help="Max scenarios to include when building the benchmark (default: all actionable).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="Force benchmark rebuild.")
    args = ap.parse_args()

    settings = get_settings()
    settings.ensure_dirs()
    init_db()

    logger.info("DB backend: %s", settings.sqlalchemy_url.split("://", 1)[0])
    logger.info("LLM backend: %s | embed: %s | reranker: %s | NLI: %s",
                settings.llm_backend, settings.embed_model,
                settings.reranker_enabled, settings.nli_enabled)

    ensure_benchmark(args.eval_n, args.build_cap, args.rebuild)
    eval_ids = select_eval_scenarios(args.eval_n, args.seed)

    configs = [(b, None) for b in BASELINE_SUITE]
    if not args.no_ablations:
        configs += ABLATIONS  # list of (label, overrides)

    # Create the experiment record.
    with session_scope() as s:
        exp = m.Experiment(
            name=f"PM-RAG benchmark campaign (N={args.eval_n})",
            research_question="RQ1-6: retrieval quality, attribution, grounding, abstention, efficiency.",
            hypothesis="H1-5: hybrid+rerank retrieval and grounding/abstention improve evidence-grounded correctness.",
            description=f"Full baseline suite + ablations over {args.eval_n} project-disjoint test scenarios.",
            status="running",
        )
        s.add(exp)
        s.commit()
        experiment_id = exp.id
    logger.info("Experiment id=%d", experiment_id)

    run_ids: list[int] = []
    t0 = time.time()
    for label, overrides in configs:
        # Ablations reuse the pm_rag_full base with overrides; baselines use their own config.
        baseline = "pm_rag_full" if overrides else label
        logger.info("---- Running '%s' (baseline=%s, overrides=%s) ----", label, baseline, overrides)
        t = time.time()
        with session_scope() as s:
            res = run_experiment_run(
                s, experiment_id=experiment_id, baseline=baseline,
                config_overrides=overrides, split="test",
                scenario_ids=eval_ids, seed=args.seed,
            )
            # Tag the run with the human-readable label for reporting.
            run = s.get(m.ExperimentRun, res["run_id"])
            run.baseline = label
            s.commit()
        run_ids.append(res["run_id"])
        logger.info("  -> run_id=%d status=%s scenarios=%d (%.1fs)",
                    res["run_id"], res["status"], res["num_scenarios"], time.time() - t)

    with session_scope() as s:
        exp = s.get(m.Experiment, experiment_id)
        exp.status = "completed"
        s.commit()

    # ---- Export + summary ----
    reports = settings.reports_path
    reports.mkdir(parents=True, exist_ok=True)
    with session_scope() as s:
        csv_text = export_runs_csv(s, run_ids)
        (reports / f"experiment_{experiment_id}_metrics.csv").write_text(csv_text, encoding="utf-8")
        matrix = compare_runs(s, run_ids)
        summaries = {rid: run_summary(s, rid) for rid in run_ids}
    (reports / f"experiment_{experiment_id}_matrix.json").write_text(
        json.dumps(matrix, indent=2), encoding="utf-8")

    # Pretty print a compact comparison for key metrics.
    key_metrics = ["recall@5", "precision@5", "ndcg@5", "mrr",
                   "attribution_f1", "supported_rate", "unsupported_rate",
                   "abstention_rate", "mean_total_ms"]
    print("\n================ EXPERIMENT SUMMARY (experiment_id=%d) ================" % experiment_id)
    header = f"{'configuration':<26}" + "".join(f"{k:>16}" for k in key_metrics)
    print(header)
    print("-" * len(header))
    for rid in run_ids:
        s = summaries[rid]
        row = {f"{r['name']}@{r['k']}" if r['k'] else r['name']: r['value'] for r in s["retrieval_metrics"]}
        row.update({r["name"]: r["value"] for r in s["recommendation_metrics"]})
        row.update({p["name"]: p["value"] for p in s["performance_metrics"]})
        line = f"{s['baseline']:<26}"
        for k in key_metrics:
            v = row.get(k)
            line += f"{(f'{v:.3f}' if isinstance(v,(int,float)) else '-'):>16}"
        print(line)
    print("-" * len(header))
    print(f"Total runs: {len(run_ids)} | wall time: {time.time()-t0:.1f}s")
    print(f"CSV:    {reports / f'experiment_{experiment_id}_metrics.csv'}")
    print(f"Matrix: {reports / f'experiment_{experiment_id}_matrix.json'}")
    print("View in UI: /research-results  and  /experiments")


if __name__ == "__main__":
    main()
