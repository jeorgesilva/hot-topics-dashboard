"""Per-topic article search via SearXNG + DuckDuckGo HTML scraping.

Used by the broad-search pipeline to find articles for each Google News RSS
headline. No API key required; SearXNG needs a self-hosted instance (set
SEARXNG_URL in .env); DuckDuckGo HTML scraping is the automatic fallback.

Core function: search_topic(query, num_results, searxng_url)
    1. SearXNG (if searxng_url provided) — primary source
    2. DuckDuckGo HTML scraping — fallback when SearXNG returns < 20 results
    Social domains and listing/category URLs are filtered before results are
    returned. Raises RuntimeError when both engines fail.
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 15  # seconds

_DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
_DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "de,en;q=0.9",
}

SOCIAL_DOMAINS: frozenset[str] = frozenset({
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "reddit.com", "tiktok.com", "youtube.com", "linkedin.com",
    "t.me", "telegram.org",
})

# URL path prefixes that indicate listing/category pages rather than individual articles.
# Checked against the first non-empty path segment only (e.g. /themen/bundesregierung → "themen").
_LISTING_PATH_PREFIXES: frozenset[str] = frozenset({
    # German listing patterns
    "themen", "thema",
    "tag", "tags",
    "kategorie", "kategorien",
    "rubrik", "rubriken",
    "schlagwort", "schlagwoerter",
    "ressort",
    "autor", "autoren",
    "archiv",
    # English listing patterns
    "topic", "topics",
    "category", "categories",
    "section", "sections",
    "author", "authors",
    "search",
    "suche",
    "newsletter",
})


def _normalize_url(url: str) -> str:
    """Remove scheme, query string and normalise trailing slash for deduplication.

    Example: 'https://www.example.com/article/?ref=1' -> 'www.example.com/article/'
    """
    parsed = urlparse(url)
    path = parsed.path or "/"
    if not path.endswith("/"):
        path += "/"
    return parsed.netloc + path


def _is_social_domain(url: str) -> bool:
    """Return True if the URL's primary domain is in SOCIAL_DOMAINS."""
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc in SOCIAL_DOMAINS


def _is_listing_url(url: str) -> bool:
    """Return True if the URL looks like a category/tag/listing page, not an individual article.

    Checks the first non-empty path segment against known listing prefixes.
    E.g. https://www.handelsblatt.com/themen/bundesregierung → "themen" → True.
    """
    path = urlparse(url).path
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False
    return segments[0].lower() in _LISTING_PATH_PREFIXES


def _searxng_search(query: str, num_results: int, base_url: str) -> list[dict]:
    """Query a SearXNG instance and return structured results.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return.
        base_url: Base URL of the SearXNG instance (e.g. http://localhost:8080).

    Returns:
        List of dicts with keys: url, title, snippet, source_engine.
        Returns empty list on any connection or parse failure.
    """
    try:
        resp = requests.get(
            f"{base_url}/search",
            params={
                "q": query,
                "format": "json",
                "categories": "news,general",
                "language": "de",
                "time_range": "month",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.debug("SearXNG unavailable: %s", exc)
        return []

    try:
        data = resp.json()
    except ValueError:
        logger.debug("SearXNG returned non-JSON response")
        return []

    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        logger.debug("SearXNG response missing 'results' field")
        return []

    results: list[dict] = []
    for r in raw_results[:num_results]:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        results.append({
            "url": url,
            "title": (r.get("title") or "").strip(),
            "snippet": (r.get("content") or "").strip(),
            "source_engine": "searxng",
        })

    logger.info("SearXNG: %d results for %r", len(results), query)
    return results


def _ddg_extract_results(soup: BeautifulSoup) -> list[dict]:
    """Extract result dicts from a parsed DuckDuckGo HTML page."""
    results: list[dict] = []
    for div in soup.select("div.result"):
        a = div.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        try:
            if href.startswith(("http://", "https://")):
                url = href
            else:
                # Legacy redirect: "//duckduckgo.com/l/?uddg=https%3A%2F%2F...&rut=..."
                parsed_href = urlparse("https:" + href)
                url = unquote(parse_qs(parsed_href.query).get("uddg", [""])[0])
        except Exception:
            continue
        if not url.startswith(("http://", "https://")):
            continue
        snippet_elem = div.select_one("a.result__snippet")
        results.append({
            "url": url,
            "title": a.get_text(strip=True),
            "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
            "source_engine": "duckduckgo",
        })
    return results


def _ddg_next_page_params(soup: BeautifulSoup) -> dict | None:
    """Extract pagination form params from DDG HTML, or None if no next page."""
    nav = soup.find("div", class_="nav-link")
    if not nav:
        return None
    form = nav.find("form")
    if not form:
        return None
    params: dict[str, str] = {}
    for inp in form.find_all("input"):
        name = inp.get("name")
        value = inp.get("value", "")
        if name:
            params[name] = value
    return params if params else None


def _ddg_html_search(query: str, num_results: int = 30) -> list[dict]:
    """Scrape DuckDuckGo HTML results for a query. No API key, no hard limit.

    Uses a persistent session to maintain cookies and follows the DDG next-page
    form to collect up to num_results results across multiple pages.

    Args:
        query: Search query string.
        num_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: url, title, snippet, source_engine.
        Returns empty list on any network or parse failure.
    """
    session = requests.Session()
    session.headers.update(_DDG_HEADERS)

    post_data: dict[str, str] = {"q": query, "b": "", "kl": "de-de", "df": "m"}
    results: list[dict] = []
    max_pages = 3

    for _page in range(max_pages):
        try:
            resp = session.post(
                _DDG_ENDPOINT,
                data=post_data,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.debug("DuckDuckGo request failed: %s", exc)
            break

        soup = BeautifulSoup(resp.text, "lxml")
        page_results = _ddg_extract_results(soup)
        results.extend(page_results)

        if len(results) >= num_results:
            break

        next_params = _ddg_next_page_params(soup)
        if not next_params or not page_results:
            break
        post_data = next_params

    logger.info("DuckDuckGo: %d results for %r", len(results), query)
    return results[:num_results]


def search_topic(
    query: str,
    num_results: int = 200,
    searxng_url: str | None = None,
) -> list[dict]:
    """Search for articles about a topic combining multiple engines.

    Strategy:
        1. SearXNG (if searxng_url provided) — primary source, up to num_results
        2. DuckDuckGo HTML scraping — fallback when SearXNG is unavailable or
           returned fewer than 20 results

    Post-processing applied to both sources:
        - Deduplication by _normalize_url
        - Social domain filtering (SOCIAL_DOMAINS)
        - Listing/category page filtering (_LISTING_PATH_PREFIXES)
        - Truncation to num_results

    Args:
        query: Search query string (use the topic label / headline).
        num_results: Maximum number of results to return.
        searxng_url: Base URL of a running SearXNG instance. If None, DDG only.

    Returns:
        List of dicts with keys: url, title, snippet, source_engine.

    Raises:
        RuntimeError: When both engines fail to return any results.
    """
    raw: list[dict] = []

    if searxng_url:
        raw = _searxng_search(query, num_results, searxng_url)

    if len(raw) < 20:
        ddg_results = _ddg_html_search(query, num_results=min(num_results, 30))
        existing_norms = {_normalize_url(r["url"]) for r in raw}
        for r in ddg_results:
            nurl = _normalize_url(r["url"])
            if nurl not in existing_norms:
                raw.append(r)
                existing_norms.add(nurl)

    if not raw:
        raise RuntimeError(
            f"Both SearXNG and DuckDuckGo failed to return results for {query!r}. "
            "Check network connectivity."
        )

    seen_norms: set[str] = set()
    filtered: list[dict] = []
    for r in raw:
        url = r.get("url", "")
        if not url:
            continue
        if _is_social_domain(url):
            continue
        if _is_listing_url(url):
            logger.debug("dropping listing/category URL: %s", url)
            continue
        nurl = _normalize_url(url)
        if nurl in seen_norms:
            continue
        seen_norms.add(nurl)
        filtered.append(r)
        if len(filtered) >= num_results:
            break

    return filtered


