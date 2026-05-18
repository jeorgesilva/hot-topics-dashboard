"""Pipeline runner — orchestrates all scrapers and writes to SQLite.

Usage:
    python -m src.scrapers.run_all
    python -m src.scrapers.run_all --sources reddit,youtube
    python -m src.scrapers.run_all --limit 10 --db-path data/test.db
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from pathlib import Path

from src.utils.db import init_db, insert_items

logger = logging.getLogger(__name__)

# Registry of available scrapers
SCRAPER_REGISTRY = {
    "reddit": {
        "module": "src.scrapers.reddit_scraper",
        "function": "scrape_reddit",
        "kwargs_key": "limit",
    },
    "youtube": {
        "module": "src.scrapers.youtube_scraper",
        "function": "scrape_youtube",
        "kwargs_key": "limit",
    },
    "google_news": {
        "module": "src.scrapers.google_rss_scraper",
        "function": "scrape_google_trends",
        "kwargs_key": "max_items",
    },
    "newsapi": {
        "module": "src.scrapers.newsapi_scraper",
        "function": "scrape_newsapi",
        "kwargs_key": "max_articles",
    },
}


def _import_scraper(source: str):
    """Dynamically import a scraper function by source name.

    Returns:
        Callable that returns list[RawItem], or None if import fails.
    """
    info = SCRAPER_REGISTRY.get(source)
    if not info:
        logger.error("Unknown source: '%s'. Available: %s", source, list(SCRAPER_REGISTRY.keys()))
        return None, None

    try:
        mod = importlib.import_module(info["module"])
        func = getattr(mod, info["function"])
        return func, info["kwargs_key"]
    except Exception:
        logger.exception("Failed to import scraper for '%s'", source)
        return None, None


def run_pipeline(
    sources: list[str] | None = None,
    limit: int = 50,
    db_path: str | Path | None = None,
) -> dict:
    """Run all scrapers, merge results, deduplicate, and write to SQLite.

    Args:
        sources: List of source names to scrape. Defaults to all.
        limit: Max items per source/subreddit/query.
        db_path: Path to SQLite database.

    Returns:
        Summary dict with counts per source and totals.
    """
    sources = sources or list(SCRAPER_REGISTRY.keys())
    conn = init_db(db_path)
    summary = {"sources": {}, "total_scraped": 0, "total_inserted": 0, "total_skipped": 0}
    all_items = []

    start = time.time()

    for source in sources:
        logger.info("=" * 50)
        logger.info("Running scraper: %s", source)
        logger.info("=" * 50)

        func, kwargs_key = _import_scraper(source)
        if func is None:
            summary["sources"][source] = {"status": "error", "items": 0}
            continue

        try:
            items = func(**{kwargs_key: limit})
            all_items.extend(items)
            summary["sources"][source] = {"status": "ok", "items": len(items)}
            logger.info("  -> %d items from %s", len(items), source)
        except Exception:
            logger.exception("Scraper '%s' failed", source)
            summary["sources"][source] = {"status": "error", "items": 0}

    # Deduplicate by ID before inserting
    seen_ids: set[str] = set()
    unique_items = []
    for item in all_items:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            unique_items.append(item)

    duplicates_in_batch = len(all_items) - len(unique_items)
    if duplicates_in_batch:
        logger.info("Removed %d cross-source duplicates", duplicates_in_batch)

    # Write to database
    inserted = insert_items(conn, unique_items)
    skipped = len(unique_items) - inserted

    elapsed = time.time() - start

    summary["total_scraped"] = len(all_items)
    summary["total_unique"] = len(unique_items)
    summary["total_inserted"] = inserted
    summary["total_skipped"] = skipped
    summary["elapsed_seconds"] = round(elapsed, 1)

    # Print summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 50)
    for src, info in summary["sources"].items():
        status_icon = "OK" if info["status"] == "ok" else "FAIL"
        logger.info("  %-12s  [%4s]  %d items", src, status_icon, info["items"])
    logger.info("-" * 50)
    logger.info("  Scraped:    %d items", summary["total_scraped"])
    logger.info("  Unique:     %d items", summary["total_unique"])
    logger.info("  Inserted:   %d new rows", summary["total_inserted"])
    logger.info("  Skipped:    %d duplicates (already in DB)", summary["total_skipped"])
    logger.info("  Time:       %.1f seconds", summary["elapsed_seconds"])
    logger.info("=" * 50)

    conn.close()
    return summary


def main():
    """CLI entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run the Hot Topics scraper pipeline",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help="Comma-separated list of sources to scrape (reddit,youtube,google_news,newsapi). Default: all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max items per source/subreddit/query. Default: 50.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database. Default: data/dashboard.db.",
    )
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None
    run_pipeline(sources=sources, limit=args.limit, db_path=args.db_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
