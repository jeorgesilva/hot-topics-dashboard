"""Tests for the Reddit RSS scraper.

All HTTP calls are mocked — no real Reddit requests during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.reddit_scraper import (
    _extract_post_id,
    _parse_reddit_atom,
    _url_to_id,
    enrich_with_comments,
    scrape_reddit,
    scrape_reddit_by_keywords,
)
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ATOM_SINGLE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>r/de</title>
  <entry>
    <title>Test Post Title</title>
    <link href="https://www.reddit.com/r/de/comments/abc123/test_post/"/>
    <updated>2026-05-01T10:00:00+00:00</updated>
    <content type="html">&lt;p&gt;Post body content here.&lt;/p&gt;</content>
  </entry>
</feed>"""

_ATOM_MULTI = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>r/germany</title>
  <entry>
    <title>First Post</title>
    <link href="https://www.reddit.com/r/germany/comments/aaa/first/"/>
    <updated>2026-05-01T09:00:00+00:00</updated>
    <content type="html">&lt;p&gt;Body A&lt;/p&gt;</content>
  </entry>
  <entry>
    <title>Second Post</title>
    <link href="https://www.reddit.com/r/germany/comments/bbb/second/"/>
    <updated>2026-05-01T08:00:00+00:00</updated>
    <content type="html">&lt;p&gt;Body B&lt;/p&gt;</content>
  </entry>
</feed>"""

_ATOM_NO_CONTENT = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>No Content Post</title>
    <link href="https://www.reddit.com/r/de/comments/xyz/no_content/"/>
    <updated>2026-05-01T10:00:00+00:00</updated>
  </entry>
</feed>"""

_ATOM_EMPTY = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""


def _mock_response(xml_text: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = xml_text
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


# ---------------------------------------------------------------------------
# _parse_reddit_atom unit tests
# ---------------------------------------------------------------------------

class TestParseRedditAtom:
    def test_parses_single_entry(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert len(items) == 1

    def test_all_rawitem_fields_present(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert set(items[0].keys()) == {
            "id", "title", "description", "source",
            "url", "platform", "timestamp", "engagement",
        }

    def test_id_is_hash_of_url(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        expected = _url_to_id("https://www.reddit.com/r/de/comments/abc123/test_post/")
        assert items[0]["id"] == expected

    def test_id_starts_with_reddit_prefix(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert items[0]["id"].startswith("reddit_")

    def test_platform_is_reddit(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert items[0]["platform"] == "reddit"

    def test_source_matches_argument(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert items[0]["source"] == "r/de"

    def test_title_extracted(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert items[0]["title"] == "Test Post Title"

    def test_description_strips_html(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert items[0]["description"] is not None
        assert "<p>" not in items[0]["description"]
        assert "Post body content here." in items[0]["description"]

    def test_missing_content_yields_none_description(self):
        items = _parse_reddit_atom(_ATOM_NO_CONTENT, "r/de")
        assert items[0]["description"] is None

    def test_timestamp_is_iso8601(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        ts = items[0]["timestamp"]
        assert "T" in ts
        assert "+" in ts or "Z" in ts or ts.endswith("+00:00")

    def test_engagement_defaults_to_zero(self):
        items = _parse_reddit_atom(_ATOM_SINGLE, "r/de")
        assert items[0]["engagement"] == {"score": 0, "comments": 0}

    def test_parses_multiple_entries(self):
        items = _parse_reddit_atom(_ATOM_MULTI, "r/germany")
        assert len(items) == 2

    def test_empty_feed_returns_empty_list(self):
        items = _parse_reddit_atom(_ATOM_EMPTY, "r/de")
        assert items == []

    def test_malformed_xml_returns_empty_list(self):
        items = _parse_reddit_atom("not xml at all <<<", "r/de")
        assert items == []

    def test_link_post_without_body_is_skipped(self):
        # URL points to external domain → link post; no <content> → skip
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Artikel auf Spiegel</title>
    <link href="https://www.spiegel.de/politik/artikel-123.html"/>
    <updated>2026-05-01T10:00:00+00:00</updated>
  </entry>
</feed>"""
        items = _parse_reddit_atom(xml, "r/de")
        assert items == []

    def test_link_post_with_body_is_kept(self):
        # URL points to external domain but has body text → keep it
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Meine Meinung dazu</title>
    <link href="https://www.spiegel.de/artikel.html"/>
    <updated>2026-05-01T10:00:00+00:00</updated>
    <content type="html">&lt;p&gt;Hier erkläre ich warum...&lt;/p&gt;</content>
  </entry>
</feed>"""
        items = _parse_reddit_atom(xml, "r/de")
        assert len(items) == 1

    def test_text_post_with_reddit_url_is_kept(self):
        # URL points to reddit.com → text post → always kept even without body
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Text post sem corpo</title>
    <link href="https://www.reddit.com/r/de/comments/xyz/post/"/>
    <updated>2026-05-01T10:00:00+00:00</updated>
  </entry>
</feed>"""
        items = _parse_reddit_atom(xml, "r/de")
        assert len(items) == 1

    def test_duplicate_urls_deduplicated(self):
        xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Post A</title>
    <link href="https://www.reddit.com/r/de/comments/dup/"/>
    <updated>2026-05-01T10:00:00+00:00</updated>
  </entry>
  <entry>
    <title>Post B</title>
    <link href="https://www.reddit.com/r/de/comments/dup/"/>
    <updated>2026-05-01T10:00:00+00:00</updated>
  </entry>
</feed>"""
        items = _parse_reddit_atom(xml, "r/de")
        assert len(items) == 1


# ---------------------------------------------------------------------------
# scrape_reddit integration tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestScrapeReddit:
    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_returns_raw_items(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_SINGLE)
        items = scrape_reddit(subreddits=["de"], limit=5)
        assert isinstance(items, list)
        assert len(items) == 1

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_source_field_uses_subreddit_name(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_SINGLE)
        items = scrape_reddit(subreddits=["de"], limit=5)
        assert items[0]["source"] == "r/de"

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_deduplicates_across_subreddits(self, mock_sleep, mock_get):
        # Both subreddits return the same URL → only one item in result
        mock_get.return_value = _mock_response(_ATOM_SINGLE)
        items = scrape_reddit(subreddits=["de", "germany"], limit=5)
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_http_error_returns_empty_gracefully(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response("", status_code=429)
        items = scrape_reddit(subreddits=["de"], limit=5)
        assert items == []

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_sort_new_uses_new_path_in_url(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit(subreddits=["de"], limit=5, sort="new")
        called_url = mock_get.call_args[0][0]
        assert "/new/" in called_url

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_sort_hot_omits_new_path(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit(subreddits=["de"], limit=5, sort="hot")
        called_url = mock_get.call_args[0][0]
        assert "/new/" not in called_url

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_sleep_called_between_subreddits(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit(subreddits=["de", "germany", "de_politik"], limit=5)
        assert mock_sleep.call_count == 2  # N-1 sleeps for N subreddits


# ---------------------------------------------------------------------------
# scrape_reddit_by_keywords tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestScrapeRedditByKeywords:
    def test_empty_keywords_returns_empty(self):
        items = scrape_reddit_by_keywords(keywords=[])
        assert items == []

    # --- global search (default: subreddits=None) ---

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_global_search_returns_raw_items(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_SINGLE)
        items = scrape_reddit_by_keywords(keywords=["Bundesrat"])
        assert isinstance(items, list)
        assert len(items) == 1

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_global_search_url_has_no_subreddit_path(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit_by_keywords(keywords=["Bundesrat"])
        called_url = mock_get.call_args[0][0]
        assert "/search.rss" in called_url
        assert "/r/" not in called_url

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_global_search_one_request_per_keyword(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit_by_keywords(keywords=["kw1", "kw2", "kw3"])
        assert mock_get.call_count == 3

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_global_search_keyword_urlencoded(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit_by_keywords(keywords=["Bundestag Wahl"])
        called_url = mock_get.call_args[0][0]
        assert "Bundestag+Wahl" in called_url or "Bundestag%20Wahl" in called_url

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_global_search_deduplicates_across_keywords(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_SINGLE)
        items = scrape_reddit_by_keywords(keywords=["Bundesrat", "Migration"])
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_global_search_http_error_graceful(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response("", status_code=503)
        items = scrape_reddit_by_keywords(keywords=["Bundesrat"])
        assert items == []

    # --- restricted search (subreddits explicitly provided) ---

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_restricted_search_url_contains_subreddit(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit_by_keywords(keywords=["Bundesrat"], subreddits=["de"])
        called_url = mock_get.call_args[0][0]
        assert "/r/de/search.rss" in called_url
        assert "restrict_sr=on" in called_url

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_restricted_search_requests_equal_keywords_times_subreddits(self, mock_sleep, mock_get):
        mock_get.return_value = _mock_response(_ATOM_EMPTY)
        scrape_reddit_by_keywords(keywords=["kw1", "kw2"], subreddits=["de", "germany"])
        # 2 keywords × 2 subreddits = 4 HTTP requests
        assert mock_get.call_count == 4


# ---------------------------------------------------------------------------
# _extract_post_id
# ---------------------------------------------------------------------------

class TestExtractPostId:
    def test_extracts_id_from_standard_url(self):
        url = "https://www.reddit.com/r/de/comments/abc123/some_title/"
        assert _extract_post_id(url) == "abc123"

    def test_extracts_id_without_trailing_title(self):
        url = "https://www.reddit.com/r/germany/comments/xyz789/"
        assert _extract_post_id(url) == "xyz789"

    def test_returns_none_for_external_url(self):
        assert _extract_post_id("https://www.spiegel.de/artikel.html") is None

    def test_returns_none_for_empty_string(self):
        assert _extract_post_id("") is None

    def test_returns_none_for_root_reddit_url(self):
        assert _extract_post_id("https://www.reddit.com/r/de/") is None


# ---------------------------------------------------------------------------
# enrich_with_comments
# ---------------------------------------------------------------------------

_COMMENTS_JSON = [
    {"data": {"children": []}},  # post listing (unused)
    {
        "data": {
            "children": [
                {"kind": "t1", "data": {"body": "Erster Kommentar zum Thema"}},
                {"kind": "t1", "data": {"body": "Zweiter Kommentar"}},
                {"kind": "more", "data": {"body": ""}},  # load-more marker — skipped
            ]
        }
    },
]


def _make_reddit_item(id: str, url: str, description: str | None = None):
    return {
        "id": id,
        "title": f"Post {id}",
        "description": description,
        "source": "r/de",
        "url": url,
        "platform": "reddit",
        "timestamp": "2026-06-01T10:00:00Z",
        "engagement": {"score": 0, "comments": 0},
    }


class TestEnrichWithComments:
    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_enriches_post_without_description(self, mock_sleep, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: _COMMENTS_JSON,
            raise_for_status=MagicMock(),
        )
        item = _make_reddit_item("p1", "https://www.reddit.com/r/de/comments/abc123/post/")
        enrich_with_comments([item])
        assert item["description"] is not None
        assert "Erster Kommentar" in item["description"]

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_skips_post_that_already_has_description(self, mock_sleep, mock_get):
        item = _make_reddit_item(
            "p2",
            "https://www.reddit.com/r/de/comments/xyz/post/",
            description="Already has text",
        )
        enrich_with_comments([item])
        mock_get.assert_not_called()
        assert item["description"] == "Already has text"

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_skips_external_url_post(self, mock_sleep, mock_get):
        # External URL → no post ID → no HTTP call
        item = _make_reddit_item("p3", "https://www.spiegel.de/artikel.html")
        enrich_with_comments([item])
        mock_get.assert_not_called()

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_graceful_on_http_error(self, mock_sleep, mock_get):
        import requests as req_lib
        mock_resp = MagicMock()
        mock_resp.status_code = 500  # non-auth error
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError()
        mock_get.return_value = mock_resp
        item = _make_reddit_item("p4", "https://www.reddit.com/r/de/comments/err1/post/")
        enrich_with_comments([item])  # must not raise
        assert item["description"] is None

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_403_bails_after_first_request(self, mock_sleep, mock_get):
        # Reddit blocked unauthenticated JSON API — must stop after first 403
        mock_get.return_value = MagicMock(status_code=403)
        items = [
            _make_reddit_item(f"p{i}", f"https://www.reddit.com/r/de/comments/id{i}/post/")
            for i in range(3)
        ]
        enrich_with_comments(items)  # must not raise
        assert mock_get.call_count == 1  # bailed after first request
        assert all(item["description"] is None for item in items)

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_skips_deleted_comments(self, mock_sleep, mock_get):
        deleted_json = [
            {"data": {"children": []}},
            {
                "data": {
                    "children": [
                        {"kind": "t1", "data": {"body": "[deleted]"}},
                        {"kind": "t1", "data": {"body": "[removed]"}},
                    ]
                }
            },
        ]
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: deleted_json,
            raise_for_status=MagicMock(),
        )
        item = _make_reddit_item("p5", "https://www.reddit.com/r/de/comments/del1/post/")
        enrich_with_comments([item])
        assert item["description"] is None

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_empty_list_makes_no_requests(self, mock_sleep, mock_get):
        enrich_with_comments([])
        mock_get.assert_not_called()

    @patch("src.scrapers.reddit_scraper.requests.get")
    @patch("src.scrapers.reddit_scraper.time.sleep")
    def test_description_truncated_to_500_chars(self, mock_sleep, mock_get):
        long_json = [
            {"data": {"children": []}},
            {
                "data": {
                    "children": [
                        {"kind": "t1", "data": {"body": "x" * 600}},
                    ]
                }
            },
        ]
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: long_json,
            raise_for_status=MagicMock(),
        )
        item = _make_reddit_item("p6", "https://www.reddit.com/r/de/comments/long1/post/")
        enrich_with_comments([item])
        assert len(item["description"]) <= 500
