"""End-to-end PM-RAG data pipeline bootstrap.

Usage (PowerShell):
    python scripts/run_pipeline.py --limit 2000

Runs: DB init -> discover+import CSVs -> schedule/budget analysis ->
scenario detection -> evidence build + indexing. A --limit restricts rows per
file (handy for a fast demo); omit it to process the full datasets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import get_logger  # noqa: E402
from app.database import init_db, session_scope  # noqa: E402
from app.ingestion import discovery, importer  # noqa: E402
from app.project_analysis import budget_analysis, schedule_analysis, scenario_detection  # noqa: E402
from app.rag import indexer  # noqa: E402

logger = get_logger("pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max rows per CSV file.")
    parser.add_argument("--index-limit", type=int, default=None, help="Max evidence chunks per dataset type.")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    init_db()

    with session_scope() as session:
        found = discovery.discover_and_classify(sample_size=200)
        logger.info("Discovered %d CSV file(s).", len(found))
        for item in found:
            if "error" in item:
                logger.warning("Skipping %s: %s", item["file"], item["error"])
                continue
            logger.info("Importing %s as %s ...", item["file"], item["dataset_type"])
            stats = importer.import_file(
                session, item["file_path"], item["dataset_type"], limit=args.limit,
                progress_cb=lambda p, t, f: logger.info("  %d/%d rows (failed=%d)", p, t, f)
                if p % 5000 == 0 else None,
            )
            logger.info("  -> %s", stats)

    with session_scope() as session:
        logger.info("Schedule analysis ...")
        logger.info("  -> %s", schedule_analysis.analyze_all(session))
        logger.info("Budget analysis ...")
        logger.info("  -> %s", budget_analysis.analyze_all(session))
        logger.info("Scenario detection ...")
        logger.info("  -> %s", scenario_detection.detect_all(session))

    if not args.skip_index:
        with session_scope() as session:
            logger.info("Evidence indexing ...")
            stats = indexer.full_reindex(
                session, limit_per_type=args.index_limit,
                progress_cb=lambda d, t: logger.info("  indexed %d/%d", d, t) if d % 2000 == 0 else None,
            )
            logger.info("  -> %s", stats)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
