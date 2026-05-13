"""Google News RSS scraper.

Fetches trending news via the Google News RSS feed:
  https://news.google.com/rss?hl={hl}&gl={geo}&ceid={geo}:{lang}

No API key needed — uses requests + stdlib XML parsing.
Each <item> in the feed is a news article; we emit one RawItem per article.
Engagement data is unavailable — fields default to {"score": 0, "comments": 0}.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.utils.models import RawItem

logger = logging.getLogger(__name__)

DEFAULT_GEO = "DE"
REQUEST_TIMEOUT = 15  # seconds

_LANG_MAP: dict[str, str] = {
    "DE": "de-DE",
    "AT": "de-AT",
    "CH": "de-CH",
    "US": "en-US",
    "GB": "en-GB",
    "AU": "en-AU",
    "FR": "fr-FR",
    "IT": "it-IT",
    "ES": "es-ES",
    "PL": "pl-PL",
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
}


def _build_rss_url(geo: str) -> str:
    """Build a Google News RSS URL for the given country code.

    Args:
        geo: ISO 3166-1 alpha-2 country code (e.g. "DE").

    Returns:
        Full RSS URL with hl, gl, and ceid parameters.
    """
    geo = geo.upper()
    hl_code = _LANG_MAP.get(geo, "en-US")
    lang = hl_code.split("-")[0]
    return f"https://news.google.com/rss?hl={hl_code}&gl={geo}&ceid={geo}:{lang}"


def _url_to_id(url: str) -> str:
    """Generate a deterministic ID from a URL."""
    return "gtrends_" + hashlib.sha256(url.encode()).hexdigest()[:12]


def _extract_domain(url: str) -> str:
    """Extract a clean domain name from a URL.

    Examples:
        "https://www.bbc.com/news/article" -> "bbc.com"
        "https://edition.cnn.com/story"    -> "edition.cnn.com"
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _parse_rss_timestamp(raw_date: str) -> str:
    """Parse an RSS pubDate string to ISO 8601.

    Google News RSS uses RFC 2822: "Mon, 11 May 2026 10:00:00 GMT".
    Falls back to current UTC time if parsing fails.
    """
    if not raw_date:
        return datetime.now(timezone.utc).isoformat()

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",  # RFC 2822 with numeric offset
        "%a, %d %b %Y %H:%M:%S %Z",  # RFC 2822 with tz abbreviation (e.g. GMT)
        "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(raw_date.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue

    return datetime.now(timezone.utc).isoformat()


def _parse_rss(xml_text: str) -> list[RawItem]:
    """Parse a Google News RSS feed and return a list of RawItems.

    Each <item> is a news article. Items without a URL are skipped.
    Duplicate URLs within a feed are deduplicated.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.exception("Failed to parse Google News RSS XML")
        return []

    channel = root.find("channel")
    if channel is None:
        logger.warning("RSS feed has no <channel> element")
        return []

    seen_urls: set[str] = set()
    items: list[RawItem] = []

    for item_elem in channel.findall("item"):
        url = (item_elem.findtext("link") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        title = (item_elem.findtext("title") or "").strip()
        pub_date = item_elem.findtext("pubDate", "")
        snippet = item_elem.findtext("description") or None

        source_elem = item_elem.find("source")
        if source_elem is not None and source_elem.text:
            source_name = source_elem.text.strip()
        else:
            source_name = _extract_domain(url)

        item: RawItem = {
            "id": _url_to_id(url),
            "title": title,
            "description": snippet,
            "source": source_name,
            "url": url,
            "platform": "google_news",
            "timestamp": _parse_rss_timestamp(pub_date),
            "engagement": {"score": 0, "comments": 0},
        }
        items.append(item)

    return items


_CSV_FIELDNAMES = [
    "id", "title", "description", "source",
    "url", "platform", "timestamp", "engagement",
]


def _update_csv(items: list[RawItem], path: Path) -> int:
    """Append new items to a CSV file, skipping rows whose id already exists.

    Creates the file (with a header row) on first run. On subsequent runs only
    rows with an id not yet present in the file are appended, so the file grows
    incrementally without duplicates.

    Args:
        items: Items to persist.
        path:  Destination CSV path. Parent directories are created if needed.

    Returns:
        Number of rows actually written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids: set[str] = set()
    is_new_file = not path.exists()

    if not is_new_file:
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing_ids.add(row["id"])

    new_items = [item for item in items if item["id"] not in existing_ids]
    if not new_items:
        return 0

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        if is_new_file:
            writer.writeheader()
        for item in new_items:
            writer.writerow({**item, "engagement": json.dumps(item["engagement"])})

    return len(new_items)


def scrape_google_trends(
    geo: str = DEFAULT_GEO,
    max_items: int = 100,
) -> list[RawItem]:
    """Fetch trending news from Google News RSS for a given country.

    Args:
        geo: ISO 3166-1 alpha-2 country code. Defaults to "DE" (Germany).
        max_items: Maximum number of RawItems to return.

    Returns:
        List of RawItem dicts, capped at max_items. Returns an empty list
        if the feed is unreachable or malformed.
    """
    url = _build_rss_url(geo)
    logger.info("Reading trending news from %s via Google News RSS...", geo)

    try:
        response = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to fetch Google News RSS for geo='%s'", geo)
        return []

    items = _parse_rss(response.text)
    items = items[:max_items]

    logger.info(
        "Google News scraper finished: %d items (geo=%s, capped_at=%d)",
        len(items),
        geo,
        max_items,
    )
    return items


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    GEO = "DE"
    results = scrape_google_trends(geo=GEO)
    csv_path = Path("data/raw") / f"google_news_{GEO.lower()}.csv"
    written = _update_csv(results, csv_path)
    for r in results[:5]:
        print(f"[{r['source']}] {r['title'][:80]}")
    print(f"... {len(results)} fetched, {written} new rows written to {csv_path}")
