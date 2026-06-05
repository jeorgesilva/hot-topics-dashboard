#!/usr/bin/env python3
"""
Hot Topics Dashboard — unified pipeline entry point.

Runs all three pipeline steps in the same process so NLP models are loaded
only once instead of twice (one per standalone script invocation).

Usage:
    python -m src.orchestrator [--broad-search] [--no-newsapi]
                                [--target-topics N] [--articles-per-topic N]
                                [--db-path PATH] [--log-level LEVEL]
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import time

from src.scoring import compute_scores, run_nlp
from src.scrapers import run_all
from src.utils.db import init_db

log = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hot Topics unified pipeline")
    p.add_argument("--target-topics", type=int, default=10)
    p.add_argument("--articles-per-topic", type=int, default=20)
    p.add_argument("--broad-search", action="store_true", default=True)
    p.add_argument("--no-broad-search", dest="broad_search", action="store_false")
    p.add_argument("--no-newsapi", action="store_true")
    p.add_argument("--db-path", default="data/dashboard.db")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def run_pipeline(args: argparse.Namespace) -> dict:
    """Execute all three pipeline steps in the same process, loading NLP models once.

    Steps:
        1. Collect    — src.scrapers.run_all
        2. NLP        — src.scoring.run_nlp (models pre-loaded via load_nlp_models)
        3. Scoring    — src.scoring.compute_scores

    Args:
        args: Namespace with target_topics, articles_per_topic, broad_search,
              no_newsapi, db_path.

    Returns:
        dict with topics_collected, topics_scored, elapsed_total_s, and
        elapsed_per_step (keys: collect, nlp, scores).
    """
    t_total = time.perf_counter()
    timings: dict[str, float] = {}

    log.info("Loading NLP models...")
    t = time.perf_counter()
    run_nlp.load_nlp_models()
    log.info("Models loaded in %.1f s", time.perf_counter() - t)

    # Step 1 — Collect
    log.info("=== Step 1/3: Collect ===")
    t = time.perf_counter()
    run_all.run_pipeline(
        target_topics=args.target_topics,
        articles_per_topic=args.articles_per_topic,
        broad_search=args.broad_search,
        skip_newsapi=args.no_newsapi,
        db_path=args.db_path,
    )
    timings["collect"] = time.perf_counter() - t

    # Step 2 — NLP (module-level model cache already warm, cold-start skipped)
    log.info("=== Step 2/3: NLP ===")
    t = time.perf_counter()
    run_nlp.run_nlp_pipeline(db_path=args.db_path)
    timings["nlp"] = time.perf_counter() - t

    # Step 3 — Composite scoring (SQLite only, no ML models)
    log.info("=== Step 3/3: Composite scoring ===")
    t = time.perf_counter()
    conn = init_db(args.db_path)
    compute_scores.score_all_topics(conn)
    conn.close()
    timings["scores"] = time.perf_counter() - t

    total = time.perf_counter() - t_total
    log.info(
        "Pipeline complete in %.1f s | collect=%.1f nlp=%.1f scores=%.1f",
        total,
        timings["collect"],
        timings["nlp"],
        timings["scores"],
    )

    result_conn = sqlite3.connect(args.db_path)
    n_topics = result_conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    n_scored = result_conn.execute(
        "SELECT COUNT(*) FROM topic_scores WHERE composite_risk IS NOT NULL"
    ).fetchone()[0]
    result_conn.close()

    return {
        "topics_collected": n_topics,
        "topics_scored": n_scored,
        "elapsed_total_s": total,
        "elapsed_per_step": timings,
    }


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    result = run_pipeline(args)
    log.info(
        "Result: %d topics collected, %d scored",
        result["topics_collected"],
        result["topics_scored"],
    )


if __name__ == "__main__":
    main()
