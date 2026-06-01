"""Full pipeline orchestrator.

Pipeline flow (default):
    1. Google News RSS (geo=DE) → trending headlines as topic seeds
    2. RSS pool: fetch articles from 29 curated German sources (trust 5–91)
    3. Reddit pool: fetch from German subreddits (optional — skipped if no credentials)
    4. NLP on each headline → generate ordered NewsAPI query variants
    5. NewsAPI (language=de) → supplement pool articles until target_count reached
    6. Persist topics + articles linked via topic_sources
    7. NLP scoring (run run_nlp.py then compute_scores.py)

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
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from src.scrapers.article_fetcher import enrich_articles_with_body
from src.scrapers.google_rss_scraper import scrape_google_trends
from src.scrapers.newsapi_scraper import NewsAPIQuotaError, scrape_newsapi
from src.scrapers.reddit_scraper import (
    enrich_with_comments,
    scrape_reddit_by_keywords,
    scrape_reddit_german,
)
from src.scrapers.rss_scraper import scrape_rss_sources
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
_INTER_QUERY_DELAY_S: float = 1.0

# Hard cap on NewsAPI calls per topic. With 15 topics × 6 calls = 90 requests/day,
# safely under the free-tier limit of 100 requests/day.
_MAX_API_CALLS_PER_TOPIC: int = 6
_MAX_REDDIT_SUPPLEMENT_CALLS: int = 3


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


def _pool_matches(article: RawItem, candidates: list[str]) -> bool:
    """Return True if article title contains at least one candidate keyword."""
    if not candidates:
        return False
    title_lower = (article.get("title") or "").lower()
    return any(c.lower() in title_lower for c in candidates)


def _build_article_pool(
    rss_days_back: int = 7,
    rss_max_per_feed: int = 20,
    reddit_limit_per_sub: int = 25,
    reddit_keywords: list[str] | None = None,
) -> tuple[dict[str, RawItem], dict[str, RawItem]]:
    """Fetch articles from curated RSS feeds and Reddit separately.

    Returns:
        (verified_pool, reddit_pool) — two dicts keyed by article id.
        verified_pool contains curated RSS articles (newsapi/rss/google_news).
        reddit_pool contains Reddit posts (platform="reddit").
    """
    verified_pool: dict[str, RawItem] = {}

    logger.info("Building article pool: RSS feeds...")
    rss_items = scrape_rss_sources(max_per_feed=rss_max_per_feed, days_back=rss_days_back)
    for item in rss_items:
        verified_pool[item["id"]] = item
    logger.info("  RSS pool: %d articles from curated sources", len(verified_pool))

    reddit_pool: dict[str, RawItem] = {}

    logger.info("Building article pool: Reddit hot feed...")
    for item in scrape_reddit_german(limit_per_sub=reddit_limit_per_sub):
        reddit_pool[item["id"]] = item
    logger.info("  Reddit hot feed: %d posts", len(reddit_pool))

    if reddit_keywords:
        logger.info(
            "Building article pool: Reddit keyword search (%d keywords)...",
            len(reddit_keywords),
        )
        before = len(reddit_pool)
        for item in scrape_reddit_by_keywords(reddit_keywords):
            if item["id"] not in reddit_pool:
                reddit_pool[item["id"]] = item
        logger.info("  Reddit keyword search: %d new posts added", len(reddit_pool) - before)

    logger.info(
        "Pool: %d verified articles, %d Reddit posts",
        len(verified_pool), len(reddit_pool),
    )
    return verified_pool, reddit_pool


def _fetch_reddit_for_topic(
    headline: str,
    reddit_pool: dict[str, RawItem],
    max_count: int = 20,
) -> list[RawItem]:
    """Draw matching Reddit posts from the pre-built Reddit pool.

    Args:
        headline: Topic seed headline used to extract candidate keywords.
        reddit_pool: Pre-built Reddit post pool from _build_article_pool.
        max_count: Maximum number of posts to return.

    Returns:
        List of RawItems from Reddit whose titles match the headline keywords.
    """
    _, candidates = _extract_candidates(headline)
    result: dict[str, RawItem] = {}
    for item in reddit_pool.values():
        if len(result) >= max_count:
            break
        if _pool_matches(item, candidates):
            result[item["id"]] = item
    return list(result.values())[:max_count]


def _fetch_articles_for_topic(
    headline: str,
    language: str,
    target_count: int,
    pool: dict[str, RawItem] | None = None,
    skip_newsapi: bool = False,
) -> list[RawItem]:
    """Fetch up to target_count verified articles for a topic.

    First draws matching articles from the pre-built verified pool (curated
    RSS sources), then supplements with NewsAPI calls until target_count is
    reached. Reddit posts are handled separately by _fetch_reddit_for_topic.

    Args:
        headline: Raw RSS headline used as the topic seed.
        language: NewsAPI language code (e.g. "de").
        target_count: Desired number of articles (e.g. 20).
        pool: Pre-built verified article pool from _build_article_pool.
        skip_newsapi: If True, only pool articles are used (no API calls).

    Returns:
        List of unique RawItem dicts, at most target_count items.
    """
    _, candidates = _extract_candidates(headline)
    collected: dict[str, RawItem] = {}

    # Phase 1: draw from pool
    if pool:
        for item in pool.values():
            if len(collected) >= target_count:
                break
            if _pool_matches(item, candidates):
                collected[item["id"]] = item
        logger.info(
            "    pool match: %d/%d articles (candidates=%s)",
            len(collected), target_count, candidates[:5],
        )

    if skip_newsapi:
        return list(collected.values())[:target_count]

    # Phase 2: supplement with NewsAPI
    variants = _generate_query_variants(headline)
    calls = 0

    for query in variants:
        if len(collected) >= target_count:
            break
        if calls >= _MAX_API_CALLS_PER_TOPIC:
            logger.info("    Reached max API calls (%d) for this topic", _MAX_API_CALLS_PER_TOPIC)
            break

        needed = target_count - len(collected)
        fetch_size = min(needed + 20, 100)

        if calls > 0:
            time.sleep(_INTER_QUERY_DELAY_S)

        try:
            articles = scrape_newsapi(
                query=query,
                language=language,
                max_articles=fetch_size,
            )
        except NewsAPIQuotaError:
            logger.error("    NewsAPI quota exhausted — aborting pipeline")
            raise
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
    rss_candidates: int = 25,
    target_topics: int = 10,
    articles_per_topic: int = 20,
    reddit_per_topic: int = 20,
    db_path: str | Path | None = None,
    rss_pool_days_back: int = 7,
    rss_pool_max_per_feed: int = 20,
    reddit_limit_per_sub: int = 25,
    skip_newsapi: bool = False,
) -> dict:
    """Run the full scrape + topic-linking pipeline (two-track).

    Collects two independent article sets per topic:
    - Verified track: articles_per_topic items from curated RSS + NewsAPI.
    - Social track:   reddit_per_topic posts from Reddit (soft limit).

    A topic qualifies when it reaches articles_per_topic verified articles.
    Reddit posts are linked as supplementary items for the social risk score.

    Args:
        geo: Google News country code (default "DE" = Germany).
        language: NewsAPI language code (default "de" = German).
        rss_candidates: How many Google RSS headlines to fetch as topic seeds.
        target_topics: Exact number of topics to publish to the dashboard.
        articles_per_topic: Required verified articles per topic (default 20).
        reddit_per_topic: Max Reddit posts to link per topic (default 20).
        db_path: SQLite database path. Defaults to data/dashboard.db.
        rss_pool_days_back: Days back to fetch for the RSS/Reddit pool.
        rss_pool_max_per_feed: Max articles per RSS feed for the pool.
        reddit_limit_per_sub: Max posts per subreddit for the pool.
        skip_newsapi: If True, skip all NewsAPI calls (pool-only mode).

    Returns:
        Summary dict with counts.
    """
    now = datetime.now(timezone.utc).isoformat()

    # ── Step 1: Google RSS → candidate seeds ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1 — Google News RSS (geo=%s, candidates=%d)", geo, rss_candidates)
    logger.info("=" * 60)
    rss_seeds = scrape_google_trends(geo=geo, max_items=rss_candidates)
    logger.info("  %d trending headlines fetched", len(rss_seeds))

    if not rss_seeds:
        logger.warning("No RSS seeds found — aborting pipeline.")
        return {"rss_seeds": 0, "topics_created": 0, "articles_inserted": 0}

    # Extract the most frequent proper nouns across all seeds for Reddit search.
    # Counting frequency across headlines surfaces the most newsworthy entities.
    kw_counter: Counter[str] = Counter()
    for seed in rss_seeds:
        proper, _ = _extract_candidates(seed["title"])
        for p in proper:
            kw_counter[p.lower()] += 1
    reddit_keywords = [kw for kw, _ in kw_counter.most_common(10)]
    logger.info("  Reddit keywords (%d): %s", len(reddit_keywords), reddit_keywords[:5])

    # ── Step 2: Build article pool from RSS + Reddit ───────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2 — Building article pool (RSS + Reddit)")
    logger.info("=" * 60)
    verified_pool, reddit_pool = _build_article_pool(
        rss_days_back=rss_pool_days_back,
        rss_max_per_feed=rss_pool_max_per_feed,
        reddit_limit_per_sub=reddit_limit_per_sub,
        reddit_keywords=reddit_keywords,
    )

    # ── Steps 3 + 4: collect qualifying topics entirely in memory ─────────────
    # Each entry: (headline, verified_articles, reddit_articles)
    qualified: list[tuple[str, list[RawItem], list[RawItem]]] = []
    topics_dropped = 0
    total_fetched = 0
    quota_exhausted = False

    logger.info("=" * 60)
    logger.info("Step 3 — Collecting %d qualifying topics", target_topics)
    logger.info("=" * 60)

    for seed_idx, seed in enumerate(rss_seeds, start=1):
        if len(qualified) >= target_topics:
            logger.info("  Target of %d topics reached — stopping early.", target_topics)
            break

        headline = seed["title"]
        variants = _generate_query_variants(headline)

        logger.info(
            "  [%2d/%2d] %s",
            seed_idx, len(rss_seeds),
            headline[:70],
        )
        logger.info(
            "           %d query variants | need %d more topics",
            len(variants), target_topics - len(qualified),
        )

        try:
            verified_articles = _fetch_articles_for_topic(
                headline=headline,
                language=language,
                target_count=articles_per_topic,
                pool=verified_pool,
                skip_newsapi=skip_newsapi,
            )
        except NewsAPIQuotaError:
            logger.error(
                "  NewsAPI quota exhausted after %d qualifying topics. "
                "Quota resets at midnight UTC. Dashboard data unchanged.",
                len(qualified),
            )
            quota_exhausted = True
            break

        total_fetched += len(verified_articles)

        if len(verified_articles) < articles_per_topic:
            topics_dropped += 1
            logger.info(
                "    ✗ dropped — only %d/%d verified articles",
                len(verified_articles), articles_per_topic,
            )
            continue

        reddit_articles = _fetch_reddit_for_topic(
            headline=headline,
            reddit_pool=reddit_pool,
            max_count=reddit_per_topic,
        )

        # Supplement from targeted keyword search when pool comes up short
        if reddit_per_topic > 0 and len(reddit_articles) < reddit_per_topic:
            _, topic_candidates = _extract_candidates(headline)
            seen_reddit = {a["id"] for a in reddit_articles}
            for kw in topic_candidates[:_MAX_REDDIT_SUPPLEMENT_CALLS]:
                if len(reddit_articles) >= reddit_per_topic:
                    break
                extra = scrape_reddit_by_keywords(
                    [kw],
                    limit_per_keyword=reddit_per_topic - len(reddit_articles) + 5,
                )
                for item in extra:
                    if len(reddit_articles) >= reddit_per_topic:
                        break
                    if item["id"] not in seen_reddit and _pool_matches(item, topic_candidates):
                        reddit_articles.append(item)
                        seen_reddit.add(item["id"])
                        reddit_pool[item["id"]] = item
            if len(reddit_articles) < reddit_per_topic:
                logger.warning(
                    "    Reddit: only %d/%d posts after supplement",
                    len(reddit_articles), reddit_per_topic,
                )

        # Fetch full article body for all verified articles
        enrich_articles_with_body(verified_articles, max_per_batch=len(verified_articles))

        # Enrich link posts that have no body text with their top comments
        enrich_with_comments(reddit_articles)

        total_fetched += len(reddit_articles)

        qualified.append((headline, verified_articles, reddit_articles))
        logger.info(
            "    ✓ qualified  verified=%d  reddit=%d  (%d/%d done)",
            len(verified_articles), len(reddit_articles),
            len(qualified), target_topics,
        )

    # ── Step 4: persist only when we have the full target set ─────────────────
    if quota_exhausted and not qualified:
        logger.warning("No topics collected — keeping existing dashboard data.")
        return {
            "rss_seeds": len(rss_seeds),
            "topics_created": 0,
            "topics_dropped": topics_dropped,
            "articles_fetched": total_fetched,
            "articles_inserted": 0,
            "quota_exhausted": True,
        }

    conn = init_db(db_path)

    conn.execute("DELETE FROM topic_sources")
    conn.execute("DELETE FROM topic_scores")
    conn.execute("DELETE FROM topics")
    conn.commit()

    articles_inserted = 0
    for topic_id, (headline, verified_articles, reddit_articles) in enumerate(qualified, start=1):
        all_articles = verified_articles + reddit_articles
        conn.execute(
            "INSERT OR REPLACE INTO topics (id, label, created_at, item_count) VALUES (?, ?, ?, ?)",
            (topic_id, headline[:255], now, len(all_articles)),
        )
        inserted = insert_items(conn, all_articles)
        articles_inserted += inserted
        conn.executemany(
            "INSERT OR IGNORE INTO topic_sources (topic_id, item_id) VALUES (?, ?)",
            [(topic_id, a["id"]) for a in all_articles],
        )
        conn.commit()
        logger.info(
            "    persisted topic_id=%d  verified=%d  reddit=%d  new=%d",
            topic_id, len(verified_articles), len(reddit_articles), inserted,
        )

    conn.close()

    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Topics published: %d / %d target", len(qualified), target_topics)
    logger.info("  Topics dropped  : %d (insufficient articles)", topics_dropped)
    logger.info("  Articles fetched: %d", total_fetched)
    logger.info("  Articles new    : %d", articles_inserted)
    logger.info("=" * 60)

    return {
        "rss_seeds": len(rss_seeds),
        "topics_created": len(qualified),
        "topics_dropped": topics_dropped,
        "articles_fetched": total_fetched,
        "articles_inserted": articles_inserted,
        "quota_exhausted": quota_exhausted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hot Topics scraper pipeline.")
    parser.add_argument("--geo", default="DE", help="Google News country code (default: DE)")
    parser.add_argument("--language", default="de", help="NewsAPI language (default: de)")
    parser.add_argument(
        "--rss-candidates", type=int, default=25,
        help="Google RSS headlines to fetch as candidates (default: 25)",
    )
    parser.add_argument(
        "--target-topics", type=int, default=10,
        help="Exact number of topics to publish (default: 10)",
    )
    parser.add_argument(
        "--articles-per-topic", type=int, default=20,
        help="Required verified articles per topic — topics that fall short are dropped (default: 20)",
    )
    parser.add_argument(
        "--reddit-per-topic", type=int, default=20,
        help="Max Reddit posts to link per topic for the social risk track (default: 20)",
    )
    parser.add_argument("--db-path", default=None, help="SQLite DB path")
    parser.add_argument(
        "--rss-pool-days-back", type=int, default=7,
        help="Days back for RSS/Reddit pool (default: 7)",
    )
    parser.add_argument(
        "--rss-pool-max-per-feed", type=int, default=20,
        help="Max articles per RSS feed for pool (default: 20)",
    )
    parser.add_argument(
        "--reddit-limit-per-sub", type=int, default=25,
        help="Max posts per subreddit for pool (default: 25)",
    )
    parser.add_argument(
        "--no-newsapi", action="store_true",
        help="Skip all NewsAPI calls — use only RSS/Reddit pool (useful when quota is exhausted)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_pipeline(
        geo=args.geo,
        language=args.language,
        rss_candidates=args.rss_candidates,
        target_topics=args.target_topics,
        articles_per_topic=args.articles_per_topic,
        reddit_per_topic=args.reddit_per_topic,
        db_path=args.db_path,
        rss_pool_days_back=args.rss_pool_days_back,
        rss_pool_max_per_feed=args.rss_pool_max_per_feed,
        reddit_limit_per_sub=args.reddit_limit_per_sub,
        skip_newsapi=args.no_newsapi,
    )


if __name__ == "__main__":
    main()
