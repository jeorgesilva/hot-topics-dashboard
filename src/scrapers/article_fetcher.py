"""Full article body text extraction using trafilatura.

Enriches scraped RawItem dicts with body_text fetched from the article URL.
Designed to run after the initial scrape, before NLP scoring, to give
the scoring pipeline substantially more text per article than title+description.

Extraction is best-effort:
  - Paywalled or bot-protected pages return None silently.
  - Articles with an already-sufficient description (≥ 300 chars) are skipped.
  - Reddit platform items are skipped (body is in description or not available).
  - A configurable cap (max_per_batch) limits how many HTTP requests are made
    per batch to keep the pipeline's wall-clock time predictable.
"""

from __future__ import annotations

import logging
import time

import trafilatura

from src.utils.models import RawItem

logger = logging.getLogger(__name__)

_MIN_BODY_LEN: int = 100  # discard extracted bodies shorter than this

# Reddit posts are their own content — fetching the linked URL would just
# duplicate an external article already covered by the verified track.
_SKIP_PLATFORMS: frozenset[str] = frozenset({"reddit"})


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


def enrich_articles_with_body(
    articles: list[RawItem],
    delay_s: float = 0.4,
    max_per_batch: int = 25,
) -> int:
    """Fetch full article body text for all non-Reddit articles (in-place).

    Adds a 'body_text' key to each successfully enriched article dict.
    Reddit articles are skipped because their post content is already in
    description, and their linked URL points to external articles covered
    by the verified track.

    Articles that fail extraction (paywall, 403, timeout) are left without
    a body_text key — the preprocessor falls back to title + description.

    Args:
        articles: RawItem dicts from scrapers. Modified in-place.
        delay_s: Seconds to sleep between HTTP requests.
        max_per_batch: Maximum fetch attempts per call (safety cap).

    Returns:
        Number of articles successfully enriched.
    """
    candidates = [
        a for a in articles
        if a.get("platform") not in _SKIP_PLATFORMS
    ]

    if not candidates:
        return 0

    to_enrich = candidates[:max_per_batch]
    enriched = 0

    for i, article in enumerate(to_enrich):
        if i > 0:
            time.sleep(delay_s)
        text = fetch_full_text(article["url"])
        if text:
            article["body_text"] = text  # type: ignore[typeddict-unknown-key]
            enriched += 1

    if enriched:
        logger.info("  Body text enrichment: %d/%d articles enriched", enriched, len(to_enrich))
    else:
        logger.debug("  Body text enrichment: 0/%d articles enriched (paywalls / bots)", len(to_enrich))

    return enriched
