"""Tests for the YouTube scraper.

All Google API calls are mocked — no real API hits during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_api_response(videos: list[dict] | None = None) -> dict:
    """Create a mock YouTube API response."""
    if videos is None:
        videos = [
            {
                "id": "dQw4w9WgXcQ",
                "snippet": {
                    "title": "Test Video Title",
                    "description": "A test video description",
                    "channelTitle": "TestChannel",
                    "publishedAt": "2026-05-11T08:00:00Z",
                },
                "statistics": {
                    "viewCount": "150000",
                    "commentCount": "320",
                },
            }
        ]
    return {"items": videos}


def _mock_youtube_client(response: dict) -> MagicMock:
    """Create a mock YouTube API client."""
    client = MagicMock()
    request = MagicMock()
    request.execute.return_value = response
    client.videos.return_value.list.return_value = request
    return client


class TestYouTubeScraper:
    """Verify YouTube scraper produces valid RawItems."""

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_returns_list_of_raw_items(self, mock_get_client):
        mock_get_client.return_value = _mock_youtube_client(_make_api_response())

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert len(items) == 1
        assert set(items[0].keys()) == {
            "id", "title", "description", "source",
            "url", "platform", "timestamp", "engagement",
        }

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_id_format(self, mock_get_client):
        mock_get_client.return_value = _mock_youtube_client(_make_api_response())

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert items[0]["id"] == "youtube_dQw4w9WgXcQ"

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_platform_is_youtube(self, mock_get_client):
        mock_get_client.return_value = _mock_youtube_client(_make_api_response())

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert items[0]["platform"] == "youtube"

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_engagement_values_are_integers(self, mock_get_client):
        mock_get_client.return_value = _mock_youtube_client(_make_api_response())

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert isinstance(items[0]["engagement"]["score"], int)
        assert isinstance(items[0]["engagement"]["comments"], int)
        assert items[0]["engagement"]["score"] == 150000
        assert items[0]["engagement"]["comments"] == 320

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_missing_statistics_default_to_zero(self, mock_get_client):
        video = {
            "id": "noStats",
            "snippet": {
                "title": "No stats video",
                "description": "",
                "channelTitle": "Chan",
                "publishedAt": "2026-05-11T08:00:00Z",
            },
            "statistics": {},
        }
        mock_get_client.return_value = _mock_youtube_client({"items": [video]})

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert items[0]["engagement"]["score"] == 0
        assert items[0]["engagement"]["comments"] == 0

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_url_format(self, mock_get_client):
        mock_get_client.return_value = _mock_youtube_client(_make_api_response())

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert items[0]["url"] == "https://youtube.com/watch?v=dQw4w9WgXcQ"

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_timestamp_is_iso8601(self, mock_get_client):
        mock_get_client.return_value = _mock_youtube_client(_make_api_response())

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert items[0]["timestamp"] == "2026-05-11T08:00:00Z"

    @patch("src.scrapers.youtube_scraper._get_client")
    def test_empty_description_becomes_none(self, mock_get_client):
        video = {
            "id": "noDesc",
            "snippet": {
                "title": "Title",
                "description": "",
                "channelTitle": "Chan",
                "publishedAt": "2026-05-11T08:00:00Z",
            },
            "statistics": {"viewCount": "1", "commentCount": "0"},
        }
        mock_get_client.return_value = _mock_youtube_client({"items": [video]})

        from src.scrapers.youtube_scraper import scrape_youtube
        items = scrape_youtube(limit=1)

        assert items[0]["description"] is None
