"""Tests for the Reddit scraper.

All PRAW calls are mocked — no real API hits during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_submission(
    id: str = "abc123",
    title: str = "Test Reddit Post",
    selftext: str = "Some body text",
    subreddit_name: str = "worldnews",
    url: str = "https://example.com/article",
    created_utc: float = 1747130400.0,  # 2025-05-13T10:00:00Z
    score: int = 250,
    num_comments: int = 83,
) -> MagicMock:
    """Create a mock PRAW Submission object."""
    sub = MagicMock()
    sub.id = id
    sub.title = title
    sub.selftext = selftext
    sub.subreddit = subreddit_name
    sub.url = url
    sub.created_utc = created_utc
    sub.score = score
    sub.num_comments = num_comments
    return sub


def _make_mock_reddit(submissions: list[MagicMock]) -> MagicMock:
    """Create a mock praw.Reddit instance that returns given submissions."""
    reddit = MagicMock()
    subreddit = MagicMock()
    subreddit.hot.return_value = iter(submissions)
    reddit.subreddit.return_value = subreddit
    return reddit


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRedditScraper:
    """Verify Reddit scraper produces valid RawItems."""

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_returns_list_of_raw_items(self, mock_get_client):
        submissions = [_make_mock_submission(id=f"post_{i}") for i in range(3)]
        mock_get_client.return_value = _make_mock_reddit(submissions)

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["worldnews"], limit=3)

        assert len(items) == 3
        for item in items:
            assert set(item.keys()) == {
                "id", "title", "description", "source",
                "url", "platform", "timestamp", "engagement",
            }

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_id_format(self, mock_get_client):
        mock_get_client.return_value = _make_mock_reddit([
            _make_mock_submission(id="xyz789")
        ])

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["politics"], limit=1)

        assert items[0]["id"] == "reddit_xyz789"

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_platform_is_reddit(self, mock_get_client):
        mock_get_client.return_value = _make_mock_reddit([_make_mock_submission()])

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["worldnews"], limit=1)

        assert items[0]["platform"] == "reddit"

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_source_includes_subreddit(self, mock_get_client):
        mock_get_client.return_value = _make_mock_reddit([
            _make_mock_submission(subreddit_name="technology")
        ])

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["technology"], limit=1)

        assert items[0]["source"] == "r/technology"

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_timestamp_is_iso8601(self, mock_get_client):
        mock_get_client.return_value = _make_mock_reddit([_make_mock_submission()])

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["worldnews"], limit=1)

        ts = items[0]["timestamp"]
        assert "T" in ts  # basic ISO 8601 check
        assert "+" in ts or "Z" in ts  # has timezone

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_engagement_has_score_and_comments(self, mock_get_client):
        mock_get_client.return_value = _make_mock_reddit([
            _make_mock_submission(score=500, num_comments=120)
        ])

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["worldnews"], limit=1)

        assert items[0]["engagement"]["score"] == 500
        assert items[0]["engagement"]["comments"] == 120

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_empty_selftext_becomes_none(self, mock_get_client):
        mock_get_client.return_value = _make_mock_reddit([
            _make_mock_submission(selftext="")
        ])

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["worldnews"], limit=1)

        assert items[0]["description"] is None

    @patch("src.scrapers.reddit_scraper._get_client")
    def test_continues_on_subreddit_error(self, mock_get_client):
        reddit = MagicMock()
        reddit.subreddit.side_effect = Exception("Subreddit not found")
        mock_get_client.return_value = reddit

        from src.scrapers.reddit_scraper import scrape_reddit
        items = scrape_reddit(subreddits=["nonexistent"], limit=1)

        assert items == []  # no crash, empty list
