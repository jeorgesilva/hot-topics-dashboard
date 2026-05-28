"""Full pipeline orchestrator.

Pipeline flow (default):
    1. Google News RSS (geo=DE) → trending headlines as topic seeds
    2. NLP on each headline → extract key terms for the NewsAPI query
    3. NewsAPI (language=de, 10 articles per topic) → articles for each topic
    4. Persist topics + articles linked via topic_sources
    5. NLP scoring (run compute_scores.py separately)

Usage:
    python -m src.scrapers.run_all
    python -m src.scrapers.run_all --max-topics 15 --articles-per-topic 5
    python -m src.scrapers.run_all --db-path data/dashboard.db
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from src.scrapers.google_rss_scraper import scrape_google_trends
from src.scrapers.newsapi_scraper import scrape_newsapi
from src.utils.db import init_db, insert_items

logger = logging.getLogger(__name__)

# German stopwords used to filter RSS headline terms before building a query.
# Keeps queries focused on named entities and subject nouns.
_DE_STOPWORDS = frozenset({
    "der", "die", "das", "ein", "eine", "einem", "einer", "eines",
    "den", "dem", "des", "und", "oder", "aber", "auch", "sich", "auf",
    "ist", "war", "sind", "wird", "hat", "haben", "von", "mit", "bei",
    "nach", "für", "aus", "an", "in", "im", "am", "zum", "zur", "zu",
    "wie", "als", "dass", "nicht", "noch", "mehr", "alle", "über",
    "vor", "unter", "zwischen", "gegen", "ohne", "durch", "nach",
    "wegen", "beim", "beim", "ihr", "seine", "ihre", "sein", "es",
    "er", "sie", "wir", "man", "dass", "wenn", "so", "bei",
})


def _query_from_title(title: str) -> str:
    """Build a focused 2-3 term NewsAPI query from a German news headline.

    Strategy:
    1. Strip punctuation and split on whitespace.
    2. Skip German stopwords and words shorter than 4 chars.
    3. Prefer proper-noun-like terms (longer, mixed-case).
    4. Return the 2 best terms so the query is broad enough to hit results.
    """
    # Remove common noise: em-dashes, curly quotes, leading symbols
    clean_title = re.sub(r"[„""»«()\[\]{}]", " ", title)
    clean_title = re.sub(r"[-–—]+", " ", clean_title)
    words = clean_title.split()

    candidates: list[str] = []
    seen: set[str] = set()
    for word in words:
        w = re.sub(r"[^\w]", "", word).strip()
        lower = w.lower()
        if len(w) >= 4 and lower not in _DE_STOPWORDS and lower not in seen:
            seen.add(lower)
            candidates.append(w)

    # Prefer capitalized words (German proper nouns / place names)
    proper = [c for c in candidates if c[0].isupper()] if candidates else []
    pool = proper if len(proper) >= 2 else candidates

    # Use the 2 most distinctive terms for a broad-but-focused query
    terms = pool[:2]
    return " ".join(terms) if terms else clean_title[:50].strip()


def run_pipeline(
    geo: str = "DE",
    language: str = "de",
    max_topics: int = 20,
    articles_per_topic: int = 10,
    db_path: str | Path | None = None,
) -> dict:
    """Run the full scrape + topic-linking pipeline.

    Args:
        geo: Google News country code (default "DE" = Germany).
        language: NewsAPI language code (default "de" = German).
        max_topics: Maximum number of trending topics from Google RSS.
        articles_per_topic: NewsAPI articles to fetch per topic.
        db_path: SQLite database path. Defaults to data/dashboard.db.

    Returns:
        Summary dict with counts.
    """
    conn = init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Google RSS → topic seeds ──────────────────────────────────────
    logger.info("=" * 55)
    logger.info("Step 1 — Google News RSS (geo=%s)", geo)
    logger.info("=" * 55)
    rss_seeds = scrape_google_trends(geo=geo, max_items=max_topics)
    logger.info("  %d trending headlines fetched", len(rss_seeds))

    if not rss_seeds:
        logger.warning("No RSS seeds found — aborting pipeline.")
        conn.close()
        return {"rss_seeds": 0, "topics_created": 0, "articles_inserted": 0}

    # Clear old topics so rankings are always fresh (child tables first)
    conn.execute("DELETE FROM topic_sources")
    conn.execute("DELETE FROM topic_scores")
    conn.execute("DELETE FROM topics")
    conn.commit()

    topics_created = 0
    articles_inserted = 0
    total_fetched = 0

    # ── Steps 2 + 3: per-topic NLP query + NewsAPI fetch ──────────────────────
    for topic_idx, seed in enumerate(rss_seeds, start=1):
        headline = seed["title"]
        query = _query_from_title(headline)

        logger.info(
            "  [%2d/%2d] %s",
            topic_idx, len(rss_seeds),
            headline[:70],
        )
        logger.info("           query → %r", query)

        # Fetch NewsAPI articles for this topic
        try:
            articles = scrape_newsapi(
                query=query,
                language=language,
                max_articles=articles_per_topic,
            )
        except Exception:
            logger.exception("    NewsAPI failed for query %r — skipping", query)
            articles = []

        total_fetched += len(articles)

        if not articles:
            logger.info("    0 articles returned — skipping topic")
            continue

        # Persist topic
        topic_id = topic_idx  # sequential integer, matches cluster_id convention
        conn.execute(
            "INSERT OR REPLACE INTO topics (id, label, created_at, item_count) VALUES (?, ?, ?, ?)",
            (topic_id, headline[:255], now, len(articles)),
        )

        # Persist articles + link to topic
        inserted = insert_items(conn, articles)
        articles_inserted += inserted

        conn.executemany(
            "INSERT OR IGNORE INTO topic_sources (topic_id, item_id) VALUES (?, ?)",
            [(topic_id, a["id"]) for a in articles],
        )
        conn.commit()

        topics_created += 1
        logger.info(
            "    ✓ topic_id=%d  articles fetched=%d  new=%d",
            topic_id, len(articles), inserted,
        )

    conn.close()

    logger.info("")
    logger.info("=" * 55)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Topics created  : %d", topics_created)
    logger.info("  Articles fetched: %d", total_fetched)
    logger.info("  Articles new    : %d", articles_inserted)
    logger.info("=" * 55)

    return {
        "rss_seeds": len(rss_seeds),
        "topics_created": topics_created,
        "articles_fetched": total_fetched,
        "articles_inserted": articles_inserted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hot Topics scraper pipeline.")
    parser.add_argument("--geo", default="DE", help="Google News country code (default: DE)")
    parser.add_argument("--language", default="de", help="NewsAPI language (default: de)")
    parser.add_argument("--max-topics", type=int, default=20, help="Max RSS topics (default: 20)")
    parser.add_argument(
        "--articles-per-topic", type=int, default=10,
        help="NewsAPI articles per topic (default: 10)",
    )
    parser.add_argument("--db-path", default=None, help="SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        geo=args.geo,
        language=args.language,
        max_topics=args.max_topics,
        articles_per_topic=args.articles_per_topic,
        db_path=args.db_path,
    )


if __name__ == "__main__":
    main()
