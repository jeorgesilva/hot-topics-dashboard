"""Diverse RSS source scraper.

Fetches recent articles from a curated list of German RSS feeds spanning the
full trust spectrum — from ARD/ZDF (trust ~90) to fringe/conspiracy sites
(trust ~5). The source list lives in config/rss_sources.csv so it can be
expanded without touching code.

No API key required. Uses requests + stdlib XML parsing.
Feeds are fetched concurrently via ThreadPoolExecutor (max_workers=20).
"""

from __future__ import annotations

import csv
import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import requests

from src.utils.csv_helpers import normalize_url
from src.utils.models import RawItem

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SOURCES_CSV = _PROJECT_ROOT / "config" / "rss_sources.csv"

REQUEST_TIMEOUT = 12  # seconds per feed

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", _HTML_TAG_RE.sub(" ", text)).strip()


def _url_to_id(url: str) -> str:
    return "rss_" + hashlib.sha256(normalize_url(url).encode()).hexdigest()[:12]


def _parse_pubdate(raw: str) -> str:
    """Parse RSS pubDate (RFC 2822 or ISO 8601) to ISO 8601 UTC string."""
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


def _parse_feed(xml_text: str, source_name: str, cutoff: datetime) -> list[RawItem]:
    """Parse a single RSS/Atom feed XML string into RawItems.

    Handles both RSS 2.0 (<item>) and Atom (<entry>) formats.
    Articles older than cutoff are discarded.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("XML parse error for source '%s'", source_name)
        return []

    # Normalise namespace prefix for Atom feeds
    ns = ""
    tag = root.tag
    if tag.startswith("{"):
        ns = tag[: tag.index("}") + 1]

    # RSS 2.0: root > channel > item
    # Atom: root (feed) > entry
    if root.tag in ("rss", "rdf:RDF") or root.find("channel") is not None:
        container = root.find("channel") or root
        entry_tag = "item"
        link_tag = "link"
        desc_tag = "description"
        date_tag = "pubDate"
    else:
        container = root
        entry_tag = f"{ns}entry"
        link_tag = f"{ns}link"
        desc_tag = f"{ns}summary"
        date_tag = f"{ns}updated"

    items: list[RawItem] = []
    seen: set[str] = set()

    for entry in container.findall(entry_tag):
        # URL — Atom uses href attribute on <link>
        link_elem = entry.find(link_tag)
        if link_elem is None:
            continue
        url = (link_elem.get("href") or link_elem.text or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)

        title_elem = entry.find(f"{ns}title" if ns else "title")
        title = _strip_html((title_elem.text or "") if title_elem is not None else "").strip()
        if not title:
            continue

        date_elem = entry.find(date_tag)
        raw_date = (date_elem.text or "") if date_elem is not None else ""
        timestamp = _parse_pubdate(raw_date)

        # Discard articles older than cutoff
        try:
            dt = datetime.fromisoformat(timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            if dt < cutoff:
                continue
        except ValueError:
            pass

        desc_elem = entry.find(desc_tag)
        description = _strip_html((desc_elem.text or "") if desc_elem is not None else "") or None

        items.append({
            "id": _url_to_id(url),
            "title": title,
            "description": description,
            "source": source_name,
            "url": url,
            "platform": "rss",
            "timestamp": timestamp,
            "engagement": {"score": 0, "comments": 0},
        })

    return items


def load_rss_sources(csv_path: Path | None = None) -> list[dict]:
    """Load the configured RSS source list.

    Args:
        csv_path: Path to the sources CSV. Defaults to config/rss_sources.csv.

    Returns:
        List of dicts with keys: name, url, trust_score, bias, language.
    """
    path = csv_path or _DEFAULT_SOURCES_CSV
    if not path.exists():
        logger.warning("RSS sources CSV not found at %s", path)
        return []
    sources = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row.get("name", "").strip()
            url = row.get("url", "").strip()
            if name and url:
                try:
                    row["trust_score"] = float(row.get("trust_score", 50))
                except ValueError:
                    row["trust_score"] = 50.0
                sources.append(row)
    return sources


def _fetch_single_feed(source: dict, days_back: int, max_per_feed: int) -> list[RawItem]:
    """Fetch and parse one RSS feed, returning up to max_per_feed items.

    All errors are caught and logged so a single dead feed never propagates
    an exception to the ThreadPoolExecutor caller.

    Args:
        source: Dict with at least 'name' and 'url' keys from rss_sources.csv.
        days_back: Discard articles older than this many days.
        max_per_feed: Hard cap on items returned from this feed.

    Returns:
        List of RawItems (may be empty on error or when feed is stale).
    """
    name = source["name"]
    feed_url = source["url"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

    try:
        resp = requests.get(feed_url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException:
        logger.warning("Failed to fetch RSS feed '%s' (%s)", name, feed_url[:60])
        return []

    items = _parse_feed(resp.text, name, cutoff)
    result = items[:max_per_feed]
    logger.info("  RSS %-30s → %d articles", name, len(result))
    return result


def scrape_rss_sources(
    max_per_feed: int = 20,
    days_back: int = 7,
    sources_csv: Path | None = None,
) -> list[RawItem]:
    """Fetch recent articles from all configured RSS sources in parallel.

    Reads source URLs from config/rss_sources.csv and fetches all feeds
    concurrently via a ThreadPoolExecutor. Errors on individual feeds are
    logged and skipped so a single dead feed never aborts the full collection.
    Deduplication by SHA-256 URL hash is applied after all workers finish.

    Args:
        max_per_feed: Maximum articles to keep per source feed.
        days_back: Discard articles older than this many days.
        sources_csv: Override path to rss_sources.csv.

    Returns:
        Combined list of RawItems from all reachable feeds, deduplicated
        by URL across all sources.
    """
    sources = load_rss_sources(sources_csv)
    if not sources:
        logger.warning("No RSS sources configured — returning empty list.")
        return []

    fetch = partial(_fetch_single_feed, days_back=days_back, max_per_feed=max_per_feed)

    all_items: list[RawItem] = []
    seen_ids: set[str] = set()

    with ThreadPoolExecutor(max_workers=20) as executor:
        for batch in executor.map(fetch, sources):
            for item in batch:
                if item["id"] not in seen_ids:
                    seen_ids.add(item["id"])
                    all_items.append(item)

    logger.info("RSS scraper finished: %d articles from %d sources", len(all_items), len(sources))
    return all_items
