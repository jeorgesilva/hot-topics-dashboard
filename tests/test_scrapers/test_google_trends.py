"""Tests for the Google News RSS scraper.

All HTTP calls are mocked — no real web requests during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _rss(items_xml: str) -> str:
    """Wrap item XML in a minimal RSS envelope."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Top stories - Google Nachrichten</title>
    {items_xml}
  </channel>
</rss>"""


def _item(
    title: str = "Bayern sichert Bundesliga-Titel",
    url: str = "https://news.google.com/rss/articles/abc123",
    source: str = "Spiegel Online",
    pub_date: str = "Mon, 11 May 2026 10:00:00 GMT",
    snippet: str = "Bayern Munich secures the title.",
) -> str:
    return f"""<item>
      <title>{title}</title>
      <link>{url}</link>
      <pubDate>{pub_date}</pubDate>
      <description>{snippet}</description>
      <source url="https://www.spiegel.de">{source}</source>
    </item>"""


def _mock_response(xml_text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = xml_text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


@patch("src.scrapers.google_rss_scraper.requests.get")
class TestGoogleNewsScraper:
    """Verify Google News RSS scraper produces valid RawItems."""

    def test_returns_list_of_raw_items(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item()))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert len(items) == 1
        assert set(items[0].keys()) == {
            "id", "title", "description", "source",
            "url", "platform", "timestamp", "engagement",
        }

    def test_platform_is_google_news(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item()))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert items[0]["platform"] == "google_news"

    def test_engagement_defaults_to_zero(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item()))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert items[0]["engagement"] == {"score": 0, "comments": 0}

    def test_source_from_source_element(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item(source="Spiegel Online")))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert items[0]["source"] == "Spiegel Online"

    def test_source_falls_back_to_domain_when_missing(self, mock_get):
        item_xml = """<item>
          <title>Article</title>
          <link>https://www.tagesschau.de/story/123</link>
          <pubDate>Mon, 11 May 2026 10:00:00 GMT</pubDate>
          <description>snippet</description>
        </item>"""
        mock_get.return_value = _mock_response(_rss(item_xml))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert items[0]["source"] == "tagesschau.de"

    def test_description_is_populated(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item(snippet="Some snippet text")))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert items[0]["description"] == "Some snippet text"

    def test_deduplicates_same_url(self, mock_get):
        same_url = "https://news.google.com/rss/articles/shared"
        two_items = _item(url=same_url, title="First") + _item(url=same_url, title="Second")
        mock_get.return_value = _mock_response(_rss(two_items))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert len(items) == 1

    def test_deterministic_id_from_url(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item()))
        from src.scrapers.google_rss_scraper import scrape_google_trends
        items1 = scrape_google_trends(geo="DE")

        mock_get.return_value = _mock_response(_rss(_item()))
        items2 = scrape_google_trends(geo="DE")

        assert items1[0]["id"] == items2[0]["id"]

    def test_id_has_gtrends_prefix(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item()))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert items[0]["id"].startswith("gtrends_")

    def test_max_items_caps_results(self, mock_get):
        two_items = _item(url="https://example.com/a") + _item(url="https://example.com/b")
        mock_get.return_value = _mock_response(_rss(two_items))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE", max_items=1)

        assert len(items) == 1

    def test_http_error_returns_empty_list(self, mock_get):
        import requests
        mock_get.side_effect = requests.HTTPError("404")

        from src.scrapers.google_rss_scraper import scrape_google_trends
        assert scrape_google_trends(geo="DE") == []

    def test_connection_error_returns_empty_list(self, mock_get):
        import requests
        mock_get.side_effect = requests.ConnectionError("timeout")

        from src.scrapers.google_rss_scraper import scrape_google_trends
        assert scrape_google_trends(geo="DE") == []

    def test_user_agent_header_is_sent(self, mock_get):
        mock_get.return_value = _mock_response(_rss(_item()))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        scrape_google_trends(geo="DE")

        _, kwargs = mock_get.call_args
        assert "User-Agent" in kwargs.get("headers", {})

    def test_multiple_items_returned(self, mock_get):
        items_xml = (
            _item(url="https://example.com/a", title="Story A")
            + _item(url="https://example.com/b", title="Story B")
            + _item(url="https://example.com/c", title="Story C")
        )
        mock_get.return_value = _mock_response(_rss(items_xml))

        from src.scrapers.google_rss_scraper import scrape_google_trends
        items = scrape_google_trends(geo="DE")

        assert len(items) == 3


class TestBuildRssUrl:
    """Unit tests for _build_rss_url."""

    def test_germany_url(self):
        from src.scrapers.google_rss_scraper import _build_rss_url
        url = _build_rss_url("DE")
        assert "hl=de-DE" in url
        assert "gl=DE" in url
        assert "ceid=DE:de" in url

    def test_us_url(self):
        from src.scrapers.google_rss_scraper import _build_rss_url
        url = _build_rss_url("US")
        assert "hl=en-US" in url
        assert "gl=US" in url
        assert "ceid=US:en" in url

    def test_geo_is_uppercased(self):
        from src.scrapers.google_rss_scraper import _build_rss_url
        url = _build_rss_url("de")
        assert "gl=DE" in url

    def test_unknown_geo_defaults_to_en_us(self):
        from src.scrapers.google_rss_scraper import _build_rss_url
        url = _build_rss_url("XX")
        assert "hl=en-US" in url
        assert "ceid=XX:en" in url


class TestParseRssTimestamp:
    """Unit tests for _parse_rss_timestamp."""

    def test_gmt_rfc2822(self):
        from src.scrapers.google_rss_scraper import _parse_rss_timestamp
        result = _parse_rss_timestamp("Mon, 11 May 2026 10:00:00 GMT")
        assert "2026-05-11" in result

    def test_numeric_offset_rfc2822(self):
        from src.scrapers.google_rss_scraper import _parse_rss_timestamp
        result = _parse_rss_timestamp("Mon, 11 May 2026 00:00:00 +0000")
        assert "2026-05-11" in result

    def test_empty_string_returns_current_time(self):
        from src.scrapers.google_rss_scraper import _parse_rss_timestamp
        result = _parse_rss_timestamp("")
        assert result and "T" in result


class TestParseRss:
    """Unit tests for _parse_rss with edge cases."""

    def test_malformed_xml_returns_empty_list(self):
        from src.scrapers.google_rss_scraper import _parse_rss
        assert _parse_rss("not xml <<<") == []

    def test_empty_channel_returns_empty_list(self):
        from src.scrapers.google_rss_scraper import _parse_rss
        xml = '<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
        assert _parse_rss(xml) == []

    def test_item_without_url_is_skipped(self):
        from src.scrapers.google_rss_scraper import _parse_rss
        xml = _rss("""<item>
          <title>No URL article</title>
          <link></link>
          <pubDate>Mon, 11 May 2026 10:00:00 GMT</pubDate>
        </item>""")
        assert _parse_rss(xml) == []
