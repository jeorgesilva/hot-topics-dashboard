"""Full article body text extraction using trafilatura.

Enriches scraped RawItem dicts with body_text fetched from the article URL.
Designed to run after the initial scrape, before NLP scoring, to give
the scoring pipeline substantially more text per article than title+description.

Extraction is best-effort: paywalled or bot-protected pages return None silently.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.parse import urlparse

import trafilatura

from src.utils.models import RawItem

logger = logging.getLogger(__name__)

_MIN_BODY_LEN: int = 300
_MAX_BODY_LEN: int = 50_000

# Paywall subscription CTAs that survive trafilatura extraction.
_PAYWALL_PHRASES: frozenset[str] = frozenset({
    "jetzt abonnieren",
    "nur für abonnenten",
    "weiterlesen mit",
    "anmelden und weiterlesen",
    "einloggen und weiterlesen",
    "exklusiv für abonnenten",
    "diesen artikel freischalten",
    "diesen artikel weiterlesen",
    "abo abschließen",
    "registrieren sie sich kostenlos",
    "zugang freischalten",
    "lesen sie weiter mit",
    "sie haben ihr kontingent",
    "subscribe to read",
    "subscribers only",
    "sign in to continue",
    "already a subscriber",
    "premium content",
})

# Matches a lowercase letter immediately followed by an uppercase letter
# within a word — a pattern impossible in natural German prose that indicates
# scrambled/obfuscated paywall text (e.g. heise.de style: "mTQOjQafznnSc").
_MIDWORD_UPPERCASE_RE = re.compile(r"[a-z][A-Z]")


def _is_paywall_text(text: str) -> bool:
    """Return True if text contains a known paywall subscription CTA."""
    lower = text.lower()
    return any(phrase in lower for phrase in _PAYWALL_PHRASES)


def _is_scrambled_text(text: str) -> bool:
    """Return True if more than 5% of long words show mid-word camelCase.

    Sites like heise.de obfuscate paywalled body text by replacing words with
    random tokens that preserve superficial structure but produce lowercase→
    uppercase transitions mid-word (e.g. 'GGYsGtrpqyzadmsjqybl').  German
    prose never does this; acronyms (BSI, KRITIS) are all-caps from position 0
    and are excluded by searching only from w[1:].
    """
    words = text.split()
    long_words = [w for w in words if len(w) > 5]
    if len(long_words) < 20:
        return False
    scrambled = sum(
        1 for w in long_words
        if _MIDWORD_UPPERCASE_RE.search(w[1:])
    )
    return (scrambled / len(long_words)) > 0.05

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


def _try_pub_date(downloaded: str) -> str | None:
    """Extract publication date from downloaded HTML via trafilatura metadata.

    Returns:
        ISO date string (YYYY-MM-DD) or None if unavailable or unparseable.
    """
    try:
        result = trafilatura.bare_extraction(downloaded, include_comments=False, include_tables=False)
        if result is None:
            return None
        date_str = getattr(result, "date", None)
        if not date_str or len(date_str) < 10:
            return None
        datetime.strptime(date_str[:10], "%Y-%m-%d")  # validate format
        return date_str[:10]
    except Exception:
        return None


def _fetch_body_for_article(article: RawItem) -> tuple[str, str | None, str | None]:
    """Fetch body text and publication date for a single article.

    Downloads once and extracts both text and date to avoid a second HTTP call.

    Args:
        article: RawItem to fetch body for.

    Returns:
        Tuple of (article_id, body_text_or_None, pub_date_or_None).
    """
    sem = _get_domain_semaphore(article["url"])
    sem.acquire()
    try:
        downloaded = trafilatura.fetch_url(article["url"])
        if not downloaded:
            return (article["id"], None, None)
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            favor_recall=True,
        )
        if text:
            text = text.strip()[:_MAX_BODY_LEN]
            if len(text) < _MIN_BODY_LEN:
                text = None
            elif _is_paywall_text(text):
                logger.debug("paywall content detected, dropping %s", article["url"])
                text = None
            elif _is_scrambled_text(text):
                logger.debug("scrambled content detected, dropping %s", article["url"])
                text = None
        pub_date = _try_pub_date(downloaded)
        return (article["id"], text, pub_date)
    except Exception as exc:
        logger.debug("unexpected error fetching %s: %s", article["url"], exc)
        return (article["id"], None, None)
    finally:
        sem.release()


def enrich_articles_with_body(articles: list[RawItem]) -> int:
    """Fetch full article body text for all articles (in-place).

    Adds a 'body_text' key to each successfully enriched article dict.
    Articles that fail extraction (paywall, 403, timeout) are left without
    a body_text key — the preprocessor falls back to title + description.

    Args:
        articles: RawItem dicts from scrapers. Modified in-place.
    """
    candidates = list(articles)

    if not candidates:
        return 0

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = {
            id_: (text, date)
            for id_, text, date in executor.map(_fetch_body_for_article, candidates)
        }

    enriched = 0
    for article in candidates:
        text, date = results.get(article["id"], (None, None))
        if text:
            article["body_text"] = text  # type: ignore[typeddict-unknown-key]
            enriched += 1
        if date:
            article["timestamp"] = date  # type: ignore[typeddict-item]

    if enriched:
        logger.info("  Body text enrichment: %d/%d articles enriched", enriched, len(candidates))
    else:
        logger.debug("  Body text enrichment: 0/%d articles enriched (paywalls / bots)", len(candidates))
    return enriched
