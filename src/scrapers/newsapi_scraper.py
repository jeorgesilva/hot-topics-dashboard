"""NewsAPI scraper (newsapi.org).

Fetches articles via the /v2/everything endpoint, then enriches each row with
full article text extracted by trafilatura. Results are persisted to a CSV that
grows incrementally — new articles are appended (deduped by URL) and text
extraction is skipped for rows that already have text.

Free tier: 100 requests/day, up to 100 articles per request, 1-month history.
Get a key at https://newsapi.org/register
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

# Allow running directly (python src/scrapers/newsapi_scraper.py) in addition
# to the normal module invocation (python -m src.scrapers.newsapi_scraper).
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import trafilatura  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.utils.models import RawItem  # noqa: E402

load_dotenv()

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

if not NEWSAPI_KEY:
    raise ValueError("NEWSAPI_KEY not found in .env file")

logger = logging.getLogger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"
DEFAULT_QUERY = "Deutschland"
DEFAULT_LANGUAGE = "de"
REQUEST_TIMEOUT = 15  # seconds

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_CSV_COLUMNS = [
    "id", "title", "description", "source",
    "url", "platform", "timestamp", "engagement", "text",
]


def _url_to_id(url: str) -> str:
    """Generate a deterministic ID from a URL."""
    return "newsapi_" + hashlib.sha256(url.encode()).hexdigest()[:12]


def _extract_domain(url: str) -> str:
    """Extract a clean domain name from a URL."""
    parsed = urlparse(url)
    domain = parsed.netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def scrape_newsapi(
    query: str = DEFAULT_QUERY,
    language: str = DEFAULT_LANGUAGE,
    max_articles: int = 50,
    sort_by: str = "publishedAt",
) -> list[RawItem]:
    """Fetch articles from NewsAPI /v2/everything.

    Args:
        query: Keyword query (e.g. "Deutschland OR Berlin"). Required by the
            /v2/everything endpoint.
        language: Two-letter language code (e.g. "de", "en").
        max_articles: Max articles to fetch. Capped at 100 (API limit).
        sort_by: Sort order — "publishedAt", "relevancy", or "popularity".

    Returns:
        List of RawItem dicts. Returns an empty list if the request fails.
    """
    params: dict = {
        "apiKey": NEWSAPI_KEY,
        "q": query,
        "language": language,
        "pageSize": min(max_articles, 100),
        "sortBy": sort_by,
    }

    logger.info(
        "Fetching NewsAPI articles (q=%s, language=%s, max=%d)",
        query, language, max_articles,
    )

    try:
        response = requests.get(NEWSAPI_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        logger.exception("Failed to fetch NewsAPI articles (q='%s')", query)
        return []

    if data.get("status") != "ok":
        logger.error(
            "NewsAPI returned status '%s': %s",
            data.get("status"), data.get("message"),
        )
        return []

    items: list[RawItem] = []
    seen_urls: set[str] = set()

    for article in data.get("articles", []):
        url = (article.get("url") or "").strip()
        title = (article.get("title") or "").strip()

        if not url or url == "https://removed.com" or url in seen_urls:
            continue
        seen_urls.add(url)

        source_name = (
            (article.get("source") or {}).get("name") or _extract_domain(url)
        )

        item: RawItem = {
            "id": _url_to_id(url),
            "title": title,
            "description": article.get("description") or None,
            "source": source_name,
            "url": url,
            "platform": "newsapi",
            "timestamp": article.get("publishedAt", ""),
            "engagement": {"score": 0, "comments": 0},
        }
        items.append(item)

    logger.info("NewsAPI scraper finished: %d articles", len(items))
    return items


def _extract_text(url: str, session: requests.Session) -> str | None:
    """Fetch a URL and extract its main text via trafilatura.

    Returns None for Google News redirect URLs (require JS) and on any error.
    """
    if "news.google.com" in url:
        return None
    try:
        resp = session.get(url, timeout=12, allow_redirects=True)
        resp.raise_for_status()
        text = trafilatura.extract(
            resp.text,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            url=url,
        )
        return text if text and text.strip() else None
    except Exception:
        logger.warning("Text extraction failed for %s", url[:70])
        return None


def fetch_full_text(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich a DataFrame with full article text via trafilatura.

    Rows that already have text and Google News redirect URLs are skipped.
    The 'text' column is added if not present.

    Args:
        df: DataFrame with at least a 'url' column.

    Returns:
        The same DataFrame with the 'text' column populated where possible.
    """
    if "text" not in df.columns:
        df["text"] = None

    session = requests.Session()
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    })

    total = len(df)
    for i, idx in enumerate(df.index):
        url = df.at[idx, "url"]

        if "news.google.com" in url:
            logger.info("[%d/%d] Skipped (Google News): %s", i + 1, total, url[:70])
            continue

        existing = df.at[idx, "text"]
        if pd.notna(existing) and str(existing).strip():
            logger.info("[%d/%d] Skipped (already has text): %s", i + 1, total, url[:70])
            continue

        text = _extract_text(url, session)
        df.at[idx, "text"] = text
        chars = len(text) if text else 0
        status = "OK" if chars > 0 else "EMPTY"
        logger.info("[%d/%d] %s (%d chars): %s", i + 1, total, status, chars, url[:70])

    return df


def _update_csv(items: list[RawItem], path: Path) -> pd.DataFrame:
    """Merge new items with an existing CSV, deduplicating by URL.

    Existing rows are placed first so already-extracted text is preserved when
    the same URL appears in both old and new results.

    Args:
        items: Freshly scraped items to merge in.
        path:  Destination CSV path. Parent directories are created if needed.

    Returns:
        Combined DataFrame (existing + new, deduped) ready for text extraction.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([
        {**item, "engagement": json.dumps(item["engagement"]), "text": None}
        for item in items
    ])

    if path.exists():
        existing_df = pd.read_csv(path)
        # Existing rows first → keep="first" preserves their text values
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["url"], keep="first")
    else:
        combined = new_df

    return combined[_CSV_COLUMNS] if set(_CSV_COLUMNS).issubset(combined.columns) else combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== FETCHING ARTICLES VIA NEWSAPI ===")
    results = scrape_newsapi(query="Deutschland", language="de", max_articles=50)

    if not results:
        print("No articles found.")
    else:
        csv_path = Path("data/raw/newsapi_de.csv")
        df = _update_csv(results, csv_path)
        new_count = len(df) - (len(pd.read_csv(csv_path)) if csv_path.exists() else 0)

        for r in results[:5]:
            print(f"[{r['source']}] {r['title'][:80]}")

        print(f"\n=== EXTRACTING ARTICLE TEXT ({len(df)} total) ===")
        df = fetch_full_text(df)

        df.to_csv(csv_path, index=False)
        print(f"\nCSV saved: {csv_path}")
        print(f"Total articles: {len(df)}")
        print(f"With text:      {df['text'].notna().sum()}")
