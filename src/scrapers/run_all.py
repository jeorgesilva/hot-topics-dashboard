"""Full pipeline orchestrator.

Pipeline flow (default):
    1. Google News RSS (geo=DE) → trending headlines as topic seeds
    2. NLP on each headline → generate ordered query variants
    3. NewsAPI (language=de) → fetch articles per variant until target_count reached
    4. Persist topics + articles linked via topic_sources
    5. NLP scoring (run run_nlp.py then compute_scores.py)

Usage:
    python -m src.scrapers.run_all
    python -m src.scrapers.run_all --max-topics 15 --articles-per-topic 100
    python -m src.scrapers.run_all --db-path data/dashboard.db
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from src.scrapers.google_rss_scraper import scrape_google_trends
from src.scrapers.newsapi_scraper import scrape_newsapi
from src.utils.db import init_db, insert_items
from src.utils.models import RawItem

logger = logging.getLogger(__name__)

# German stopwords — keeps queries focused on named entities and subject nouns.
_DE_STOPWORDS = frozenset({
    "der", "die", "das", "ein", "eine", "einem", "einer", "eines",
    "den", "dem", "des", "und", "oder", "aber", "auch", "sich", "auf",
    "ist", "war", "sind", "wird", "hat", "haben", "von", "mit", "bei",
    "nach", "für", "aus", "an", "in", "im", "am", "zum", "zur", "zu",
    "wie", "als", "dass", "nicht", "noch", "mehr", "alle", "über",
    "vor", "unter", "zwischen", "gegen", "ohne", "durch", "nach",
    "wegen", "beim", "ihr", "seine", "ihre", "sein", "es",
    "er", "sie", "wir", "man", "wenn", "so",
    # common news filler words
    "sagt", "gibt", "geht", "zeigt", "kommt", "macht", "soll", "will",
    "neue", "neuen", "neuer", "neuem", "neues", "jetzt", "erst", "schon",
    "beim", "kein", "keine", "keinen", "keiner", "keines",
})

# Delay between consecutive NewsAPI calls for the same topic (free tier safety).
_INTER_QUERY_DELAY_S: float = 0.5

# Hard cap on NewsAPI calls per topic. With 15 topics × 6 calls = 90 requests/day,
# safely under the free-tier limit of 100 requests/day.
_MAX_API_CALLS_PER_TOPIC: int = 6


def _extract_candidates(title: str) -> tuple[list[str], list[str]]:
    """Return (proper_nouns, all_candidates) from a German headline.

    proper_nouns: subset of candidates that start with an uppercase letter.
    all_candidates: words ≥4 chars, not in stopwords, deduplicated.
    """
    clean = re.sub(r"[„""»«()\[\]{}]", " ", title)
    clean = re.sub(r"[-–—]+", " ", clean)

    candidates: list[str] = []
    seen: set[str] = set()
    for word in clean.split():
        w = re.sub(r"[^\w]", "", word).strip()
        lower = w.lower()
        if len(w) >= 4 and lower not in _DE_STOPWORDS and lower not in seen:
            seen.add(lower)
            candidates.append(w)

    proper = [c for c in candidates if c[0].isupper()] if candidates else []
    return proper, candidates


def _generate_query_variants(title: str) -> list[str]:
    """Return an ordered list of NewsAPI query strings for a German headline.

    Variants are ordered from most specific (proper-noun pairs) to broadest
    (single terms), so early calls return the most relevant articles.

    Priority order:
        1. All 2-word pairs from proper nouns (most specific)
        2. All 2-word pairs mixing one proper noun + one other candidate
        3. All 2-word pairs from remaining candidates
        4. Individual proper nouns (broadest — used as a last resort)
        5. Individual other candidates
    """
    proper, candidates = _extract_candidates(title)
    others = [c for c in candidates if c not in proper]

    seen_queries: set[str] = set()
    variants: list[str] = []

    def _add(q: str) -> None:
        q = q.strip()
        if q and q not in seen_queries:
            seen_queries.add(q)
            variants.append(q)

    # 1. Proper-noun pairs
    for a, b in combinations(proper, 2):
        _add(f"{a} {b}")

    # 2. Proper × other candidate pairs
    for p in proper:
        for o in others:
            _add(f"{p} {o}")

    # 3. All remaining candidate pairs
    for a, b in combinations(candidates, 2):
        _add(f"{a} {b}")

    # 4. Single proper nouns
    for p in proper:
        _add(p)

    # 5. Single other candidates
    for o in others:
        _add(o)

    # Fallback: first 50 chars of original title
    if not variants:
        _add(title[:50].strip())

    return variants


def _fetch_articles_for_topic(
    headline: str,
    language: str,
    target_count: int,
) -> list[RawItem]:
    """Fetch up to target_count unique articles for a topic.

    Iterates through query variants generated from the headline, accumulating
    unique articles (deduped by id) until target_count is reached or all
    variants are exhausted (up to _MAX_API_CALLS_PER_TOPIC calls).

    Args:
        headline: Raw RSS headline used as the topic seed.
        language: NewsAPI language code (e.g. "de").
        target_count: Desired number of articles (e.g. 100).

    Returns:
        List of unique RawItem dicts, at most target_count items.
    """
    variants = _generate_query_variants(headline)
    collected: dict[str, RawItem] = {}
    calls = 0

    for query in variants:
        if len(collected) >= target_count:
            break
        if calls >= _MAX_API_CALLS_PER_TOPIC:
            logger.info("    Reached max API calls (%d) for this topic", _MAX_API_CALLS_PER_TOPIC)
            break

        needed = target_count - len(collected)
        # Fetch slightly more than needed to compensate for dedup losses.
        fetch_size = min(needed + 20, 100)

        if calls > 0:
            time.sleep(_INTER_QUERY_DELAY_S)

        try:
            articles = scrape_newsapi(
                query=query,
                language=language,
                max_articles=fetch_size,
            )
        except Exception:
            logger.exception("    NewsAPI failed for query %r — skipping", query)
            calls += 1
            continue

        calls += 1
        new_count = 0
        for a in articles:
            if a["id"] not in collected:
                collected[a["id"]] = a
                new_count += 1

        logger.info(
            "    query=%r → %d returned, %d new (total: %d/%d)",
            query, len(articles), new_count, len(collected), target_count,
        )

        # No point trying more variants if this one returned nothing
        if not articles:
            continue

    result = list(collected.values())[:target_count]
    if len(result) < target_count:
        logger.warning(
            "    Could only find %d/%d articles after %d API calls",
            len(result), target_count, calls,
        )
    return result


def run_pipeline(
    geo: str = "DE",
    language: str = "de",
    max_topics: int = 20,
    articles_per_topic: int = 100,
    db_path: str | Path | None = None,
) -> dict:
    """Run the full scrape + topic-linking pipeline.

    Args:
        geo: Google News country code (default "DE" = Germany).
        language: NewsAPI language code (default "de" = German).
        max_topics: Maximum number of trending topics from Google RSS.
        articles_per_topic: Target articles per topic. The fetcher retries with
            alternative queries until this count is reached or API call budget
            is exhausted (default 100).
        db_path: SQLite database path. Defaults to data/dashboard.db.

    Returns:
        Summary dict with counts.
    """
    conn = init_db(db_path)
    now = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Google RSS → topic seeds ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1 — Google News RSS (geo=%s)", geo)
    logger.info("=" * 60)
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

    # ── Steps 2 + 3: per-topic query variants + NewsAPI fetch ─────────────────
    for topic_idx, seed in enumerate(rss_seeds, start=1):
        headline = seed["title"]
        variants = _generate_query_variants(headline)

        logger.info(
            "  [%2d/%2d] %s",
            topic_idx, len(rss_seeds),
            headline[:70],
        )
        logger.info("           %d query variants generated", len(variants))

        articles = _fetch_articles_for_topic(
            headline=headline,
            language=language,
            target_count=articles_per_topic,
        )

        total_fetched += len(articles)

        if not articles:
            logger.info("    0 articles found — skipping topic")
            continue

        # Persist topic
        topic_id = topic_idx
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
            "    ✓ topic_id=%d  articles=%d  new=%d",
            topic_id, len(articles), inserted,
        )

    conn.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Topics created  : %d", topics_created)
    logger.info("  Articles fetched: %d", total_fetched)
    logger.info("  Articles new    : %d", articles_inserted)
    logger.info("=" * 60)

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
        "--articles-per-topic", type=int, default=100,
        help="Target articles per topic — retries with alternative queries until reached (default: 100)",
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
