"""Tests for the NewsAPI scraper.

All HTTP and trafilatura calls are mocked — no real requests during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest


def _make_api_response(
    articles: list[dict] | None = None,
    status: str = "ok",
) -> dict:
    """Build a minimal NewsAPI /v2/everything response."""
    if articles is None:
        articles = [
            {
                "source": {"id": "spiegel-online", "name": "Spiegel Online"},
                "title": "Bayern sichert Bundesliga-Titel",
                "description": "Bayern Munich hat die Meisterschaft gesichert.",
                "url": "https://www.spiegel.de/sport/fussball/article-1",
                "publishedAt": "2026-05-13T10:00:00Z",
            }
        ]
    return {"status": status, "totalResults": len(articles), "articles": articles}


def _mock_response(data: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


@patch("src.scrapers.newsapi_scraper.requests.get")
class TestScrapeNewsapi:
    """Verify scrape_newsapi() produces valid RawItems."""

    def test_returns_list_of_raw_items(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        items = scrape_newsapi()

        assert len(items) == 1
        assert set(items[0].keys()) == {
            "id", "title", "description", "source",
            "url", "platform", "timestamp", "engagement",
        }

    def test_platform_is_newsapi(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi()[0]["platform"] == "newsapi"

    def test_id_has_newsapi_prefix(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi()[0]["id"].startswith("newsapi_")

    def test_source_name_from_api(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi()[0]["source"] == "Spiegel Online"

    def test_source_falls_back_to_domain(self, mock_get):
        article = {
            "source": {"id": None, "name": None},
            "title": "Article",
            "description": "desc",
            "url": "https://www.tagesschau.de/story/123",
            "publishedAt": "2026-05-13T10:00:00Z",
        }
        mock_get.return_value = _mock_response(_make_api_response([article]))

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi()[0]["source"] == "tagesschau.de"

    def test_engagement_defaults_to_zero(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi()[0]["engagement"] == {"score": 0, "comments": 0}

    def test_removed_articles_are_skipped(self, mock_get):
        articles = [
            {"source": {"name": "X"}, "title": "Gone",
             "description": None, "url": "https://removed.com", "publishedAt": ""},
            {"source": {"name": "Y"}, "title": "Real",
             "description": "d", "url": "https://example.com/real", "publishedAt": "2026-05-13T10:00:00Z"},
        ]
        mock_get.return_value = _mock_response(_make_api_response(articles))

        from src.scrapers.newsapi_scraper import scrape_newsapi
        items = scrape_newsapi()
        assert len(items) == 1
        assert items[0]["title"] == "Real"

    def test_articles_without_url_are_skipped(self, mock_get):
        articles = [
            {"source": {"name": "X"}, "title": "No URL",
             "description": None, "url": "", "publishedAt": ""},
            {"source": {"name": "Y"}, "title": "Has URL",
             "description": "d", "url": "https://example.com/a", "publishedAt": "2026-05-13T10:00:00Z"},
        ]
        mock_get.return_value = _mock_response(_make_api_response(articles))

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert len(scrape_newsapi()) == 1

    def test_duplicate_urls_deduplicated(self, mock_get):
        articles = [
            {"source": {"name": "A"}, "title": "First",
             "description": "d", "url": "https://example.com/same", "publishedAt": ""},
            {"source": {"name": "B"}, "title": "Second",
             "description": "d", "url": "https://example.com/same", "publishedAt": ""},
        ]
        mock_get.return_value = _mock_response(_make_api_response(articles))

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert len(scrape_newsapi()) == 1

    def test_api_error_status_returns_empty(self, mock_get):
        mock_get.return_value = _mock_response(
            {"status": "error", "code": "apiKeyInvalid", "message": "Invalid key."}
        )

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi() == []

    def test_http_error_returns_empty(self, mock_get):
        import requests
        mock_get.side_effect = requests.HTTPError("401")

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi() == []

    def test_connection_error_returns_empty(self, mock_get):
        import requests
        mock_get.side_effect = requests.ConnectionError("timeout")

        from src.scrapers.newsapi_scraper import scrape_newsapi
        assert scrape_newsapi() == []

    def test_query_passed_to_api(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        scrape_newsapi(query="Bundesliga OR Bayern")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["q"] == "Bundesliga OR Bayern"

    def test_language_passed_to_api(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        scrape_newsapi(language="en")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["language"] == "en"

    def test_sort_by_passed_to_api(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        scrape_newsapi(sort_by="popularity")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["sortBy"] == "popularity"

    def test_max_articles_capped_at_100(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())

        from src.scrapers.newsapi_scraper import scrape_newsapi
        scrape_newsapi(max_articles=200)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["pageSize"] == 100

    def test_deterministic_id_from_url(self, mock_get):
        mock_get.return_value = _mock_response(_make_api_response())
        from src.scrapers.newsapi_scraper import scrape_newsapi
        id1 = scrape_newsapi()[0]["id"]

        mock_get.return_value = _mock_response(_make_api_response())
        id2 = scrape_newsapi()[0]["id"]

        assert id1 == id2


class TestFetchFullText:
    """Verify fetch_full_text() enriches DataFrame rows correctly."""

    @patch("src.scrapers.newsapi_scraper.trafilatura.extract", return_value="Full article text here.")
    @patch("src.scrapers.newsapi_scraper.requests.Session")
    def test_adds_text_column(self, mock_session_cls, mock_extract):
        mock_resp = MagicMock()
        mock_resp.text = "<html>content</html>"
        mock_resp.raise_for_status = MagicMock()
        mock_session_cls.return_value.__enter__ = MagicMock()
        session = MagicMock()
        session.get.return_value = mock_resp
        mock_session_cls.return_value = session

        from src.scrapers.newsapi_scraper import fetch_full_text
        df = pd.DataFrame([{"url": "https://example.com/article", "text": None}])
        result = fetch_full_text(df)

        assert "text" in result.columns
        assert result.at[0, "text"] == "Full article text here."

    @patch("src.scrapers.newsapi_scraper.trafilatura.extract")
    @patch("src.scrapers.newsapi_scraper.requests.Session")
    def test_skips_google_news_urls(self, mock_session_cls, mock_extract):
        session = MagicMock()
        mock_session_cls.return_value = session

        from src.scrapers.newsapi_scraper import fetch_full_text
        df = pd.DataFrame([{
            "url": "https://news.google.com/rss/articles/abc123",
            "text": None,
        }])
        fetch_full_text(df)

        session.get.assert_not_called()
        mock_extract.assert_not_called()

    @patch("src.scrapers.newsapi_scraper.trafilatura.extract")
    @patch("src.scrapers.newsapi_scraper.requests.Session")
    def test_skips_rows_with_existing_text(self, mock_session_cls, mock_extract):
        session = MagicMock()
        mock_session_cls.return_value = session

        from src.scrapers.newsapi_scraper import fetch_full_text
        df = pd.DataFrame([{
            "url": "https://example.com/article",
            "text": "Already extracted text.",
        }])
        fetch_full_text(df)

        session.get.assert_not_called()

    @patch("src.scrapers.newsapi_scraper.trafilatura.extract")
    @patch("src.scrapers.newsapi_scraper.requests.Session")
    def test_handles_request_error_gracefully(self, mock_session_cls, mock_extract):
        import requests as req
        session = MagicMock()
        session.get.side_effect = req.ConnectionError("timeout")
        mock_session_cls.return_value = session

        from src.scrapers.newsapi_scraper import fetch_full_text
        df = pd.DataFrame([{"url": "https://example.com/article", "text": None}])
        result = fetch_full_text(df)

        assert pd.isna(result.at[0, "text"])


class TestUpdateCsv:
    """Verify _update_csv() appends and deduplicates correctly."""

    def test_creates_csv_and_returns_count(self, tmp_path):
        from src.scrapers.newsapi_scraper import _update_csv
        items = [
            {"id": "newsapi_abc", "title": "T", "description": None, "source": "S",
             "url": "https://example.com/1", "platform": "newsapi",
             "timestamp": "2026-05-13T10:00:00Z", "engagement": {"score": 0, "comments": 0}},
        ]
        path = tmp_path / "test.csv"
        written = _update_csv(items, path)

        assert written == 1
        assert path.exists()

    def test_deduplicates_by_id(self, tmp_path):
        from src.scrapers.newsapi_scraper import _update_csv
        item = {
            "id": "newsapi_abc", "title": "T", "description": None, "source": "S",
            "url": "https://example.com/1", "platform": "newsapi",
            "timestamp": "2026-05-13T10:00:00Z", "engagement": {"score": 0, "comments": 0},
        }
        path = tmp_path / "test.csv"

        _update_csv([item], path)
        written_again = _update_csv([item], path)

        assert written_again == 0

    def test_preserves_existing_text(self, tmp_path):
        import pandas as pd
        from src.scrapers.newsapi_scraper import _update_csv
        item = {
            "id": "newsapi_abc", "title": "T", "description": None, "source": "S",
            "url": "https://example.com/1", "platform": "newsapi",
            "timestamp": "2026-05-13T10:00:00Z", "engagement": {"score": 0, "comments": 0},
        }
        path = tmp_path / "test.csv"

        # First run — write item, then simulate text extraction by editing the file
        _update_csv([item], path)
        df = pd.read_csv(path)
        df["text"] = df["text"].astype(object)  # all-NaN col is float64 by default
        df.at[0, "text"] = "Extracted body text."
        df.to_csv(path, index=False)

        # Second run — same ID is skipped, so the row with text is untouched
        written = _update_csv([item], path)
        assert written == 0
        df2 = pd.read_csv(path)
        assert df2.at[0, "text"] == "Extracted body text."
