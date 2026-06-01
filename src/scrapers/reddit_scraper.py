"""Reddit scraper using native RSS/Atom feeds — no credentials required.

Reddit generates public Atom feeds for subreddits and keyword searches:
  Subreddit feed  : https://www.reddit.com/r/{sub}/.rss?limit=N
  Subreddit new   : https://www.reddit.com/r/{sub}/new/.rss?limit=N
  Keyword search  : https://www.reddit.com/r/{sub}/search.rss?q={kw}&restrict_sr=on

No PRAW, no OAuth, no API keys — plain HTTP GET on public feed URLs.
Integrate with keyword seeds from the pipeline's Google RSS step to surface
Reddit discussions directly related to trending topics.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import requests

from src.utils.models import RawItem

logger = logging.getLogger(__name__)

# German-language subreddits spanning mainstream discussion to Q&A and fringe
DEFAULT_SUBREDDITS: list[str] = [
    "de",           # general German news & discussion (~600k members)
    "germany",      # expat/international perspective on Germany
    "nachrichten",  # German news aggregator
    "FragReddit",   # German Q&A — text posts, diverse opinions
    "AskGermany",   # questions about Germany — text posts
]

_RSS_BASE = "https://www.reddit.com"
_REQUEST_TIMEOUT = 15
_INTER_REQUEST_DELAY_S: float = 1.0

_ATOM_NS = "{http://www.w3.org/2005/Atom}"

_HEADERS = {
    "User-Agent": "hot-topics-dashboard/1.0 RSS reader (non-commercial research)",
    "Accept": "application/atom+xml, application/rss+xml, application/xml, */*",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Matches the full string when it's only submission metadata
_REDDIT_BOILERPLATE_RE = re.compile(
    r"^\s*submitted\s+by\s+/u/\S+(\s+\[link\])?(\s+\[comments\])?\s*$"
)
# Strips trailing submission metadata from the end of a description
_BOILERPLATE_SUFFIX_RE = re.compile(
    r"\s*submitted\s+by\s+/u/\S+.*$", re.DOTALL
)
# Extracts the base36 post ID from a reddit.com/r/.../comments/{id}/... URL
_REDDIT_POST_ID_RE = re.compile(r"reddit\.com/r/\w+/comments/([a-z0-9]+)/")


def is_reddit_available() -> bool:
    """Always True — RSS feeds require no credentials."""
    return True


def _strip_html(text: str) -> str:
    clean = _HTML_TAG_RE.sub(" ", text)
    clean = html.unescape(clean)
    return _WHITESPACE_RE.sub(" ", clean).strip()


def _is_boilerplate(text: str) -> bool:
    """Return True if text is only Reddit link-post submission metadata."""
    return bool(_REDDIT_BOILERPLATE_RE.match(text))


def _url_to_id(url: str) -> str:
    return "reddit_" + hashlib.sha256(url.encode()).hexdigest()[:12]


def _parse_date(raw: str) -> str:
    """Parse ISO 8601 date string to ISO 8601 UTC."""
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


def _parse_reddit_atom(xml_text: str, source: str) -> list[RawItem]:
    """Parse a Reddit Atom feed XML string into RawItems.

    Args:
        xml_text: Raw Atom response body from a Reddit feed URL.
        source: Human-readable label for the source (e.g. "r/de").

    Returns:
        List of RawItems; empty list on parse error.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("XML parse error for Reddit feed '%s'", source)
        return []

    items: list[RawItem] = []
    seen_urls: set[str] = set()

    for entry in root.findall(f"{_ATOM_NS}entry"):
        link_elem = entry.find(f"{_ATOM_NS}link")
        if link_elem is None:
            continue
        url = (link_elem.get("href") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        title_elem = entry.find(f"{_ATOM_NS}title")
        title = _strip_html((title_elem.text or "") if title_elem is not None else "").strip()
        if not title:
            continue

        # Use per-entry subreddit label when available (global search results
        # each belong to a different community)
        category_elem = entry.find(f"{_ATOM_NS}category")
        entry_source = (
            category_elem.get("label") or source
            if category_elem is not None else source
        )

        updated_elem = entry.find(f"{_ATOM_NS}updated")
        timestamp = _parse_date((updated_elem.text or "") if updated_elem is not None else "")

        content_elem = entry.find(f"{_ATOM_NS}content")
        description: str | None = None
        if content_elem is not None:
            raw_content = _strip_html(content_elem.text or "")
            raw_content = _BOILERPLATE_SUFFIX_RE.sub("", raw_content).strip()
            if raw_content and not _is_boilerplate(raw_content):
                description = raw_content[:500]

        # Skip link posts: URL points to an external domain and there is no
        # body text. These are already covered by the RSS/NewsAPI pipeline and
        # add nothing for NLP scoring beyond a duplicate title.
        is_text_post = "reddit.com" in url
        if not is_text_post and not description:
            continue

        items.append({
            "id": _url_to_id(url),
            "title": title,
            "description": description,
            "source": entry_source,
            "url": url,
            "platform": "reddit",
            "timestamp": timestamp,
            "engagement": {"score": 0, "comments": 0},
        })

    return items


def _fetch_feed(url: str, source: str) -> list[RawItem]:
    """Fetch and parse one Reddit Atom feed URL.

    Returns an empty list on any HTTP or parsing error — never raises.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_reddit_atom(resp.text, source)
    except requests.RequestException as exc:
        logger.warning("Failed to fetch Reddit feed '%s': %s", source, exc)
        return []


def scrape_reddit(
    subreddits: list[str] | None = None,
    limit: int = 25,
    sort: str = "hot",
) -> list[RawItem]:
    """Fetch posts from subreddit feeds via RSS (no credentials required).

    Args:
        subreddits: Subreddit names (without "r/"). Defaults to DEFAULT_SUBREDDITS.
        limit: Max posts to fetch per subreddit.
        sort: Feed sort — "hot" or "new".

    Returns:
        Combined deduplicated list of RawItems.
    """
    subreddits = subreddits or DEFAULT_SUBREDDITS
    items: list[RawItem] = []
    seen_ids: set[str] = set()

    sort_segment = "/new" if sort == "new" else ""

    for i, sub in enumerate(subreddits):
        if i > 0:
            time.sleep(_INTER_REQUEST_DELAY_S)

        url = f"{_RSS_BASE}/r/{sub}{sort_segment}/.rss?limit={limit}"
        source = f"r/{sub}"
        feed_items = _fetch_feed(url, source)

        added = 0
        for item in feed_items[:limit]:
            if item["id"] not in seen_ids:
                seen_ids.add(item["id"])
                items.append(item)
                added += 1

        logger.info("  Reddit %-30s → %d posts", source, added)

    logger.info("Reddit scraper: %d posts from %d subreddits", len(items), len(subreddits))
    return items


def scrape_reddit_by_keywords(
    keywords: list[str],
    subreddits: list[str] | None = None,
    limit_per_keyword: int = 25,
    time_filter: str = "week",
) -> list[RawItem]:
    """Search Reddit for posts matching keywords via RSS.

    By default performs a **global search** across all public subreddits
    (one request per keyword). Pass a list of subreddit names to restrict
    the search to those communities instead.

    Args:
        keywords: Search terms (typically proper nouns from headline seeds).
        subreddits: Subreddits to search within. None (default) = global search
            across all public subreddits — finds the most popular posts
            regardless of which community they came from.
        limit_per_keyword: Max posts returned per keyword query.
        time_filter: Reddit time range — "day", "week", "month", "year", "all".

    Returns:
        Combined deduplicated list of RawItems across all queries.
    """
    if not keywords:
        return []

    items: list[RawItem] = []
    seen_ids: set[str] = set()
    request_count = 0

    for keyword in keywords:
        if request_count > 0:
            time.sleep(_INTER_REQUEST_DELAY_S)

        encoded_kw = urllib.parse.quote_plus(keyword)

        if subreddits is None:
            # Global search — all public subreddits, one request per keyword
            url = (
                f"{_RSS_BASE}/search.rss"
                f"?q={encoded_kw}&sort=relevance&t={time_filter}&limit={limit_per_keyword}"
            )
            feed_items = _fetch_feed(url, "reddit/global")
            request_count += 1

            new_count = 0
            for item in feed_items[:limit_per_keyword]:
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    items.append(item)
                    new_count += 1

            if new_count:
                logger.debug("  Reddit global [%r] → %d new posts", keyword, new_count)
        else:
            # Restricted search within specified subreddits
            for sub in subreddits:
                time.sleep(_INTER_REQUEST_DELAY_S)
                url = (
                    f"{_RSS_BASE}/r/{sub}/search.rss"
                    f"?q={encoded_kw}&restrict_sr=on&sort=relevance"
                    f"&t={time_filter}&limit={limit_per_keyword}"
                )
                feed_items = _fetch_feed(url, f"r/{sub}")
                request_count += 1

                new_count = 0
                for item in feed_items[:limit_per_keyword]:
                    if item["id"] not in seen_ids:
                        seen_ids.add(item["id"])
                        items.append(item)
                        new_count += 1

                if new_count:
                    logger.debug("  Reddit search [%r @ %s] → %d new posts", keyword, sub, new_count)

    scope = "global" if subreddits is None else f"{len(subreddits)} subreddits"
    logger.info(
        "Reddit keyword search (%s): %d posts — %d keywords, %d requests",
        scope, len(items), len(keywords), request_count,
    )
    return items


def scrape_reddit_german(limit_per_sub: int = 25, sort: str = "hot") -> list[RawItem]:
    """Fetch hot/new posts from all default German-language subreddits.

    Args:
        limit_per_sub: Posts per subreddit (default 25).
        sort: Feed sort — "hot" or "new".

    Returns:
        Combined list of RawItems.
    """
    return scrape_reddit(subreddits=DEFAULT_SUBREDDITS, limit=limit_per_sub, sort=sort)


# ---------------------------------------------------------------------------
# Comment enrichment
# ---------------------------------------------------------------------------

def _extract_post_id(url: str) -> str | None:
    """Return the base36 post ID from a reddit.com comments URL, or None."""
    m = _REDDIT_POST_ID_RE.search(url)
    return m.group(1) if m else None


class _RedditAuthError(Exception):
    """Reddit JSON API returned 401/403/429 — authentication required."""


def _fetch_post_comments(url: str, max_comments: int = 5) -> str | None:
    """Fetch top-level comments for a Reddit post via the public JSON API.

    Args:
        url: The reddit.com discussion URL (must contain /comments/{id}/).
        max_comments: Maximum number of top-level comments to concatenate.

    Returns:
        Pipe-separated comment bodies (up to 500 chars), or None on failure.

    Raises:
        _RedditAuthError: If the API returns 401, 403, or 429 — signals that
            the caller should stop retrying (OAuth credentials required).
    """
    post_id = _extract_post_id(url)
    if not post_id:
        return None
    json_url = (
        f"https://www.reddit.com/comments/{post_id}/.json"
        f"?limit={max_comments}&depth=1&sort=top"
    )
    try:
        resp = requests.get(json_url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
        if resp.status_code in (401, 403, 429):
            raise _RedditAuthError(f"HTTP {resp.status_code} from Reddit JSON API")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        children = data[1].get("data", {}).get("children", [])
        texts: list[str] = []
        for child in children[:max_comments]:
            if child.get("kind") != "t1":
                continue
            body = child.get("data", {}).get("body", "")
            if body and body not in ("[deleted]", "[removed]"):
                texts.append(body.strip())
        return (" | ".join(texts))[:500] if texts else None
    except _RedditAuthError:
        raise
    except Exception:
        return None


def enrich_with_comments(
    items: list[RawItem],
    max_comments: int = 5,
    delay_s: float = 1.0,
) -> None:
    """Fetch top comments for Reddit posts that have no body text (in-place).

    Only posts whose URL contains a Reddit discussion path are enriched —
    external-URL link posts are skipped because their post ID is not available
    from the feed URL alone.

    Bails out immediately and silently if Reddit's JSON API returns an auth
    error (403/429) — this endpoint requires OAuth since 2023. In that case
    NLP analysis falls back to post titles only.

    Args:
        items: RawItems (platform='reddit'). Items that already have a
               non-empty description are left unchanged.
        max_comments: Top-level comments to concatenate per post.
        delay_s: Inter-request delay to respect Reddit's rate limit.
    """
    to_enrich = [
        item for item in items
        if not item.get("description") and _extract_post_id(item.get("url", ""))
    ]
    if not to_enrich:
        return
    logger.debug("Attempting comment enrichment for %d Reddit posts...", len(to_enrich))
    enriched = 0
    for i, item in enumerate(to_enrich):
        if i > 0:
            time.sleep(delay_s)
        try:
            text = _fetch_post_comments(item["url"], max_comments=max_comments)
        except _RedditAuthError:
            logger.debug(
                "Reddit JSON API requires OAuth — comment enrichment unavailable. "
                "Add REDDIT_CLIENT_ID/SECRET to .env and use PRAW to enable this feature."
            )
            return
        if text:
            item["description"] = text
            enriched += 1
    if enriched:
        logger.info("  Comment enrichment: %d/%d posts enriched", enriched, len(to_enrich))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Hot feed ===")
    results = scrape_reddit_german(limit_per_sub=5)
    for r in results[:3]:
        print(f"[{r['source']}] {r['title'][:80]}")
    print(f"Total: {len(results)} posts\n")

    print("=== Keyword search ===")
    kw_results = scrape_reddit_by_keywords(["Bundesrat", "Migration"], limit_per_query=3)
    for r in kw_results[:3]:
        print(f"[{r['source']}] {r['title'][:80]}")
    print(f"Total: {len(kw_results)} posts")
