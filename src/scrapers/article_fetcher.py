"""Full article body text extraction using trafilatura.

Enriches scraped RawItem dicts with body_text fetched from the article URL.
Designed to run after the initial scrape, before NLP scoring, to give
the scoring pipeline substantially more text per article than title+description.

Extraction is best-effort: paywalled or bot-protected pages return None silently.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import trafilatura

from src.utils.models import RawItem

logger = logging.getLogger(__name__)

_MIN_BODY_LEN: int = 100

_domain_semaphores: dict[str, threading.Semaphore] = {}
_semaphores_lock = threading.Lock()


def _get_domain_semaphore(url: str, max_concurrent: int = 2) -> threading.Semaphore:
    domain = urlparse(url).netloc
    with _semaphores_lock:
        if domain not in _domain_semaphores:
            _domain_semaphores[domain] = threading.Semaphore(max_concurrent)
    return _domain_semaphores[domain]


def fetch_full_text(url: str) -> str | None:
    """Download a URL and extract the main article body via trafilatura.

    Args:
        url: Article URL to fetch and parse.

    Returns:
        Extracted plain text, or None if extraction failed or text is too short.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        if text and len(text.strip()) >= _MIN_BODY_LEN:
            return text.strip()
        return None
    except Exception as exc:
        logger.debug("body fetch failed for %s: %s", url, exc)
        return None


def _fetch_body_for_article(article: RawItem) -> tuple[str, str | None]:
    """Fetch body text for a single article with per-domain rate limiting.

    Args:
        article: RawItem to fetch body for.

    Returns:
        Tuple of (article_id, body_text_or_None).
    """
    sem = _get_domain_semaphore(article["url"])
    sem.acquire()
    try:
        return (article["id"], fetch_full_text(article["url"]))
    except Exception as exc:
        logger.debug("unexpected error fetching %s: %s", article["url"], exc)
        return (article["id"], None)
    finally:
        sem.release()


def enrich_articles_with_body(articles: list[RawItem]) -> None:
    """Fetch full article body text for all articles (in-place).

    Adds a 'body_text' key to each successfully enriched article dict.
    Articles that fail extraction (paywall, 403, timeout) are left without
    a body_text key — the preprocessor falls back to title + description.

    Args:
        articles: RawItem dicts from scrapers. Modified in-place.
    """
    candidates = list(articles)

    if not candidates:
        return

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = dict(executor.map(_fetch_body_for_article, candidates))

    enriched = 0
    for article in candidates:
        body = results.get(article["id"])
        if body:
            article["body_text"] = body  # type: ignore[typeddict-unknown-key]
            enriched += 1

    if enriched:
        logger.info("  Body text enrichment: %d/%d articles enriched", enriched, len(candidates))
    else:
        logger.debug("  Body text enrichment: 0/%d articles enriched (paywalls / bots)", len(candidates))
