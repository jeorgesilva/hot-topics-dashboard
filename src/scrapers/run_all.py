"""Full scrape + topic-discovery pipeline.

Default mode (--broad-search, on by default):
    1. Google News RSS top 10 trending headlines as topic queries
    2. SearXNG/DDG search per topic; social/listing filter; global dedup
    3. Trafilatura full-text extraction with text-quality filtering
       (length, paywall phrases, scrambled-text detection) then persist

Curated mode (--no-broad-search):
    1. Google News RSS topic seed headlines
    2. Curated RSS pool article candidates
    3. NewsAPI supplement per topic (unless --no-newsapi)
    4. Qualify then persist

Usage:
    python -m src.scrapers.run_all
    python -m src.scrapers.run_all --articles-per-topic 15
    python -m src.scrapers.run_all --no-broad-search --db-path data/dashboard.db
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

from src.scrapers.article_fetcher import enrich_articles_with_body
from src.scrapers.broad_search import search_topic
from src.scrapers.google_rss_scraper import scrape_google_trends
from src.scrapers.newsapi_scraper import NewsAPIQuotaError, scrape_newsapi
from src.scrapers.rss_scraper import scrape_rss_sources
from src.utils.db import complete_run, init_db, insert_items, start_run
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

_AGE_CUTOFF_DAYS: int = 14


def _extract_candidates(title: str) -> tuple[list[str], list[str]]:
    """Return (proper_nouns, all_candidates) from a German headline.

    proper_nouns: subset of candidates that start with an uppercase letter.
    all_candidates: words ≥4 chars, not in stopwords, deduplicated.
    """
    clean = re.sub(r'[„""»«()\[\]{}]', " ", title)
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
) -> dict[str, RawItem]:
    """Fetch articles from curated RSS feeds.

    Returns:
        Mapping of article id → RawItem from the 45 curated German feeds.
    """
    verified_pool: dict[str, RawItem] = {}
    logger.info("Building article pool: RSS feeds...")
    rss_items = scrape_rss_sources(max_per_feed=rss_max_per_feed, days_back=rss_days_back)
    for item in rss_items:
        verified_pool[item["id"]] = item
    logger.info("  RSS pool: %d articles from curated sources", len(verified_pool))
    return verified_pool


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
    reached.

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


def _filter_by_age(articles: list[RawItem], max_days: int = _AGE_CUTOFF_DAYS) -> list[RawItem]:
    """Drop articles whose publication date is more than max_days calendar days ago.

    Compares the date portion of each article's timestamp against today's date.
    Articles with an unparseable timestamp are kept (benefit of the doubt).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=max_days)).date()
    kept: list[RawItem] = []
    dropped = 0
    for article in articles:
        ts = str(article.get("timestamp", ""))[:10]
        try:
            pub_date = datetime.strptime(ts, "%Y-%m-%d").date()
            if pub_date >= cutoff:
                kept.append(article)
            else:
                dropped += 1
        except ValueError:
            kept.append(article)
    if dropped:
        logger.info(
            "    age filter: %d/%d articles dropped (older than %d days)",
            dropped, len(articles), max_days,
        )
    return kept


def _search_results_to_raw_items(results: list[dict]) -> list[RawItem]:
    """Convert search result dicts from broad_search into RawItem format."""
    now = datetime.now(timezone.utc).isoformat()
    items: list[RawItem] = []
    for r in results:
        url = r["url"]
        items.append({
            "id": "broad_" + hashlib.sha256(url.encode()).hexdigest()[:12],
            "title": r.get("title") or url,
            "description": r.get("snippet") or None,
            "source": urlparse(url).netloc,
            "url": url,
            "platform": "broad_search",
            "timestamp": now,
            "engagement": {"score": 0, "comments": 0},
        })
    return items


def run_pipeline(
    geo: str = "DE",
    language: str = "de",
    rss_candidates: int = 50,
    target_topics: int = 10,
    articles_per_topic: int = 20,
    db_path: str | Path | None = None,
    rss_pool_days_back: int = 7,
    rss_pool_max_per_feed: int = 20,
    skip_newsapi: bool = False,
    min_articles_no_newsapi: int = 10,
    broad_search: bool = False,
) -> dict:
    """Run the full scrape + topic-linking pipeline.

    Collects verified articles per topic from curated RSS + NewsAPI
    (or broad_search when --broad-search is set).

    A topic qualifies when it reaches the minimum verified article count.
    With NewsAPI enabled that minimum is articles_per_topic (default 20).
    With --no-newsapi the minimum falls back to min_articles_no_newsapi
    (default 10) because the RSS pool alone rarely covers niche stories.

    Args:
        geo: Google News country code (default "DE" = Germany).
        language: NewsAPI language code (default "de" = German).
        rss_candidates: How many Google RSS headlines to fetch as topic seeds.
        target_topics: Exact number of topics to publish to the dashboard.
        articles_per_topic: Target verified articles per topic (default 20).
        db_path: SQLite database path. Defaults to data/dashboard.db.
        rss_pool_days_back: Days back to fetch for the RSS pool.
        rss_pool_max_per_feed: Max articles per RSS feed for the pool.
        skip_newsapi: If True, skip all NewsAPI calls (pool-only mode).
        min_articles_no_newsapi: Minimum verified articles to qualify a topic
            when skip_newsapi=True (default 10, lower than articles_per_topic
            because the RSS pool alone cannot supply 20 for niche stories).
        broad_search: If True, use dynamic web search instead of the curated
            RSS pool. SearXNG is used when SEARXNG_URL is set, otherwise DDG.

    Returns:
        Summary dict with counts.
    """
    now = datetime.now(timezone.utc).isoformat()

    min_qualify = min_articles_no_newsapi if (skip_newsapi or broad_search) else articles_per_topic

    # Open DB once here so we can register the run before any scraping starts.
    conn_early = init_db(db_path)
    run_id = start_run(conn_early)
    conn_early.close()

    if broad_search:
        # ── Step 1: Google News RSS → top 10 headlines as topic queries ───────
        logger.info("=" * 60)
        logger.info("Step 1 — Google News RSS topics (geo=%s, top 10)", geo)
        logger.info("=" * 60)
        rss_seeds = scrape_google_trends(geo=geo, max_items=10)
        logger.info("  %d trending headlines fetched", len(rss_seeds))

        if not rss_seeds:
            logger.warning("No RSS seeds found — aborting pipeline.")
            conn_done = init_db(db_path)
            complete_run(conn_done, run_id)
            conn_done.close()
            return {
                "rss_seeds": 0, "topics_created": 0, "topics_dropped": 0,
                "articles_fetched": 0, "articles_inserted": 0, "quota_exhausted": False,
            }

        searxng_url = os.getenv("SEARXNG_URL") or None
        global_seen_urls: set[str] = set()
        qualified: list[tuple[str, list[RawItem]]] = []
        total_fetched = 0

        # ── Steps 2–3: per-topic search + scrape + text-quality filter ────────
        logger.info("=" * 60)
        logger.info("Steps 2–3 — Per-topic search, scrape, and text-quality filter")
        logger.info("=" * 60)

        for seed in rss_seeds:
            headline = seed["title"]
            logger.info("  Topic: %s", headline[:80])

            try:
                results = search_topic(headline, num_results=50, searxng_url=searxng_url)
            except RuntimeError as exc:
                logger.warning("    Search failed: %s", exc)
                continue

            # Deduplicate URLs globally across all topics.
            articles: list[RawItem] = []
            for item in _search_results_to_raw_items(results):
                if item["url"] not in global_seen_urls:
                    global_seen_urls.add(item["url"])
                    articles.append(item)

            total_fetched += len(articles)
            logger.info("    %d articles after global dedup", len(articles))

            if not articles:
                logger.info("    skipped — all URLs already seen in earlier topics")
                continue

            # Scrape all articles; paywall/scramble/length filters run inside
            # article_fetcher._fetch_body_for_article and drop bad text silently.
            enrich_articles_with_body(articles)
            articles = _filter_by_age(articles)
            articles = [a for a in articles if a.get("body_text")]

            if not articles:
                logger.info("    skipped — no articles with usable body text")
                continue

            kept = articles[:articles_per_topic]
            qualified.append((headline, kept))
            logger.info("    ✓ %d articles with body text", len(kept))

        # ── Persist ───────────────────────────────────────────────────────────
        conn = init_db(db_path)
        articles_inserted = 0

        for headline, topic_articles in qualified:
            cur = conn.execute(
                "INSERT INTO topics (label, created_at, item_count, run_id) VALUES (?, ?, ?, ?)",
                (headline[:255], now, len(topic_articles), run_id),
            )
            topic_db_id = cur.lastrowid
            inserted = insert_items(conn, topic_articles)
            articles_inserted += inserted
            conn.executemany(
                "INSERT OR IGNORE INTO topic_sources (topic_id, item_id) VALUES (?, ?)",
                [(topic_db_id, a["id"]) for a in topic_articles],
            )
            conn.commit()
            logger.info(
                "    persisted topic_id=%d  run_id=%d  articles=%d  new=%d",
                topic_db_id, run_id, len(topic_articles), inserted,
            )

        complete_run(conn, run_id)
        conn.close()

        topics_dropped = len(rss_seeds) - len(qualified)
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE (RSS-driven broad search)")
        logger.info("  Topics published: %d / %d RSS seeds", len(qualified), len(rss_seeds))
        logger.info("  Topics skipped  : %d (no usable articles)", topics_dropped)
        logger.info("  Articles fetched: %d", total_fetched)
        logger.info("  Articles new    : %d", articles_inserted)
        logger.info("=" * 60)

        return {
            "rss_seeds": len(rss_seeds),
            "topics_created": len(qualified),
            "topics_dropped": topics_dropped,
            "articles_fetched": total_fetched,
            "articles_inserted": articles_inserted,
            "quota_exhausted": False,
        }

    # ── Step 1: Google RSS → candidate seeds ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1 — Google News RSS (geo=%s, candidates=%d)", geo, rss_candidates)
    logger.info("=" * 60)
    rss_seeds = scrape_google_trends(geo=geo, max_items=rss_candidates)
    logger.info("  %d trending headlines fetched", len(rss_seeds))

    if not rss_seeds:
        logger.warning("No RSS seeds found — aborting pipeline.")
        return {"rss_seeds": 0, "topics_created": 0, "articles_inserted": 0}

    # ── Step 2: Build article pool from RSS ───────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2 — Building article pool (RSS)")
    logger.info("=" * 60)
    verified_pool = _build_article_pool(
        rss_days_back=rss_pool_days_back,
        rss_max_per_feed=rss_pool_max_per_feed,
    )

    # ── Steps 3 + 4: collect qualifying topics entirely in memory ─────────────
    qualified: list[tuple[str, list[RawItem]]] = []
    topics_dropped = 0
    total_fetched = 0
    quota_exhausted = False

    logger.info("=" * 60)
    logger.info(
        "Step 3 — Collecting %d qualifying topics (min verified: %d)",
        target_topics, min_qualify,
    )
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

        if len(verified_articles) < min_qualify:
            topics_dropped += 1
            logger.info(
                "    ✗ dropped — only %d/%d verified articles",
                len(verified_articles), min_qualify,
            )
            continue

        # Fetch full article body for all verified articles
        enrich_articles_with_body(verified_articles)
        verified_articles = _filter_by_age(verified_articles)

        if len(verified_articles) < min_qualify:
            topics_dropped += 1
            logger.info(
                "    ✗ dropped after age filter — only %d/%d articles",
                len(verified_articles), min_qualify,
            )
            continue

        qualified.append((headline, verified_articles))
        logger.info(
            "    ✓ qualified  verified=%d  (%d/%d done)",
            len(verified_articles),
            len(qualified), target_topics,
        )

    # ── Step 4: persist only when we have the full target set ─────────────────
    if quota_exhausted and not qualified:
        logger.warning("No topics collected — keeping existing dashboard data.")
        conn_fail = init_db(db_path)
        complete_run(conn_fail, run_id)
        conn_fail.close()
        return {
            "rss_seeds": len(rss_seeds),
            "topics_created": 0,
            "topics_dropped": topics_dropped,
            "articles_fetched": total_fetched,
            "articles_inserted": 0,
            "quota_exhausted": True,
        }

    conn = init_db(db_path)

    articles_inserted = 0
    for headline, verified_articles in qualified:
        cur = conn.execute(
            "INSERT INTO topics (label, created_at, item_count, run_id) VALUES (?, ?, ?, ?)",
            (headline[:255], now, len(verified_articles), run_id),
        )
        topic_db_id = cur.lastrowid
        inserted = insert_items(conn, verified_articles)
        articles_inserted += inserted
        conn.executemany(
            "INSERT OR IGNORE INTO topic_sources (topic_id, item_id) VALUES (?, ?)",
            [(topic_db_id, a["id"]) for a in verified_articles],
        )
        conn.commit()
        logger.info(
            "    persisted topic_id=%d  run_id=%d  verified=%d  new=%d",
            topic_db_id, run_id, len(verified_articles), inserted,
        )

    complete_run(conn, run_id)
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
        "--rss-candidates", type=int, default=50,
        help="Google RSS headlines to fetch as candidates (default: 50)",
    )
    parser.add_argument(
        "--target-topics", type=int, default=10,
        help="Exact number of topics to publish (default: 10)",
    )
    parser.add_argument(
        "--articles-per-topic", type=int, default=20,
        help="Required verified articles per topic — topics that fall short are dropped (default: 20)",
    )
    parser.add_argument("--db-path", default=None, help="SQLite DB path")
    parser.add_argument(
        "--rss-pool-days-back", type=int, default=7,
        help="Days back for RSS pool (default: 7)",
    )
    parser.add_argument(
        "--rss-pool-max-per-feed", type=int, default=20,
        help="Max articles per RSS feed for pool (default: 20)",
    )
    parser.add_argument(
        "--no-newsapi", action="store_true",
        help="Skip all NewsAPI calls — use only RSS pool (useful when quota is exhausted)",
    )
    parser.add_argument(
        "--min-articles-no-newsapi", type=int, default=10,
        help=(
            "Minimum verified articles to qualify a topic when --no-newsapi is set "
            "(default: 15; lower than --articles-per-topic because the RSS pool alone "
            "cannot supply 20 articles for niche stories)"
        ),
    )
    parser.add_argument(
        "--broad-search", action="store_true", default=True,
        help="Use dynamic topic search (SearXNG/DDG) instead of the curated RSS pool (default: on)",
    )
    parser.add_argument(
        "--no-broad-search", dest="broad_search", action="store_false",
        help="Disable broad search and use curated RSS pool only (useful for tests)",
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
        db_path=args.db_path,
        rss_pool_days_back=args.rss_pool_days_back,
        rss_pool_max_per_feed=args.rss_pool_max_per_feed,
        skip_newsapi=args.no_newsapi,
        min_articles_no_newsapi=args.min_articles_no_newsapi,
        broad_search=args.broad_search,
    )


if __name__ == "__main__":
    main()
