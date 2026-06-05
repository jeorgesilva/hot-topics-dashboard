"""Tests for src/scrapers/broad_search.py.

All HTTP calls are mocked — no live network requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.broad_search import (
    SOCIAL_DOMAINS,
    _is_social_domain,
    _normalize_url,
    search_topic,
)


# ---------------------------------------------------------------------------
# _normalize_url
# ---------------------------------------------------------------------------

def test_normalize_url_strips_scheme_and_query():
    assert _normalize_url("https://www.example.com/article/?ref=1") == "www.example.com/article/"


def test_normalize_url_adds_trailing_slash():
    assert _normalize_url("https://example.com/article") == "example.com/article/"


def test_normalize_url_strips_fragment():
    assert _normalize_url("https://example.com/page#section") == "example.com/page/"


# ---------------------------------------------------------------------------
# _is_social_domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "https://twitter.com/foo",
    "https://www.twitter.com/foo",
    "https://x.com/bar",
    "https://www.facebook.com/page",
    "https://reddit.com/r/de",
    "https://youtube.com/watch?v=123",
    "https://t.me/channel",
])
def test_is_social_domain_blocks_social(url: str):
    assert _is_social_domain(url) is True


@pytest.mark.parametrize("url", [
    "https://tagesschau.de/news",
    "https://www.spiegel.de/politik",
    "https://faz.net/aktuell",
    "https://bbc.com/world",
])
def test_is_social_domain_allows_news(url: str):
    assert _is_social_domain(url) is False


# ---------------------------------------------------------------------------
# search_topic — mocked network
# ---------------------------------------------------------------------------

def _make_ddg_response(urls: list[str]) -> MagicMock:
    """Build a mock DDG HTML response with the given URLs as direct hrefs."""
    links = "".join(
        f'<div class="result">'
        f'<a class="result__a" href="{url}">Title {i}</a>'
        f'<a class="result__snippet">Snippet {i}</a>'
        f'</div>'
        for i, url in enumerate(urls)
    )
    html = f"<html><body>{links}</body></html>"
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = html
    mock_resp.raise_for_status.return_value = None
    return mock_resp


def _news_urls(n: int) -> list[str]:
    return [f"https://newssite{i}.de/article/{i}" for i in range(n)]


@patch("src.scrapers.broad_search.requests.Session")
def test_search_topic_returns_deduplicated_results(mock_session_cls):
    urls = _news_urls(5)
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post.return_value = _make_ddg_response(urls)

    results = search_topic("Test Query", num_results=10)

    assert len(results) == 5
    assert all(r["source_engine"] == "duckduckgo" for r in results)
    # All returned URLs are in the mock set
    assert {r["url"] for r in results} == set(urls)


@patch("src.scrapers.broad_search.requests.Session")
def test_search_topic_filters_social_domains(mock_session_cls):
    urls = _news_urls(3) + [
        "https://twitter.com/user/status/1",
        "https://facebook.com/post/2",
    ]
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post.return_value = _make_ddg_response(urls)

    results = search_topic("Test Query", num_results=10)

    returned_urls = {r["url"] for r in results}
    assert "https://twitter.com/user/status/1" not in returned_urls
    assert "https://facebook.com/post/2" not in returned_urls
    assert len(results) == 3


@patch("src.scrapers.broad_search.requests.Session")
def test_search_topic_deduplicates_same_url(mock_session_cls):
    # DDG returning the same URL twice
    urls = ["https://spiegel.de/article/1"] * 3 + _news_urls(2)
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post.return_value = _make_ddg_response(urls)

    results = search_topic("Test Query", num_results=10)

    assert len(results) == len({r["url"] for r in results})


@patch("src.scrapers.broad_search.requests.Session")
def test_search_topic_raises_when_both_fail(mock_session_cls):
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.post.return_value = _make_ddg_response([])

    with pytest.raises(RuntimeError, match="SearXNG and DuckDuckGo failed"):
        search_topic("Test Query", num_results=10)


@patch("src.scrapers.broad_search._ddg_html_search")
@patch("src.scrapers.broad_search._searxng_search")
def test_search_topic_uses_ddg_fallback_when_searxng_returns_few(
    mock_searxng, mock_ddg
):
    mock_searxng.return_value = [
        {"url": f"https://a.de/{i}", "title": "t", "snippet": "", "source_engine": "searxng"}
        for i in range(5)  # fewer than 20 — triggers DDG fallback
    ]
    mock_ddg.return_value = [
        {"url": f"https://b.de/{i}", "title": "t", "snippet": "", "source_engine": "duckduckgo"}
        for i in range(10)
    ]

    results = search_topic("Query", num_results=20, searxng_url="http://localhost:8080")

    assert mock_searxng.called
    assert mock_ddg.called
    assert len(results) == 15  # 5 searxng + 10 ddg (no overlap)
