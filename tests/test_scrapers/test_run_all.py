"""Tests for the pipeline runner (run_all.py).

Verifies orchestration, error handling, and deduplication logic.
All scraper imports are mocked.
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest

from src.utils.models import RawItem


def _make_item(id: str, platform: str = "reddit") -> RawItem:
    return {
        "id": id,
        "title": f"Title for {id}",
        "description": None,
        "source": "test",
        "url": f"https://example.com/{id}",
        "platform": platform,
        "timestamp": "2026-05-11T10:00:00Z",
        "engagement": {"score": 1, "comments": 0},
    }


class TestRunPipeline:
    """Verify the pipeline runner orchestrates correctly."""

    @patch("src.scrapers.reddit_scraper.scrape_reddit")
    @patch("src.scrapers.youtube_scraper.scrape_youtube")
    @patch("src.scrapers.google_rss_scraper.scrape_google_trends")
    def test_runs_all_sources(self, mock_gn, mock_yt, mock_reddit, tmp_path):
        mock_reddit.return_value = [_make_item("r1")]
        mock_yt.return_value = [_make_item("y1", "youtube")]
        mock_gn.return_value = [_make_item("g1", "google_news")]

        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(sources=["reddit", "youtube", "google_news"], db_path=tmp_path / "test.db")

        assert summary["total_scraped"] == 3
        assert summary["total_inserted"] == 3
        assert summary["sources"]["reddit"]["status"] == "ok"
        assert summary["sources"]["youtube"]["status"] == "ok"
        assert summary["sources"]["google_news"]["status"] == "ok"

    @patch("src.scrapers.reddit_scraper.scrape_reddit")
    @patch("src.scrapers.youtube_scraper.scrape_youtube")
    @patch("src.scrapers.google_rss_scraper.scrape_google_trends")
    def test_deduplicates_across_sources(self, mock_gn, mock_yt, mock_reddit, tmp_path):
        # Same ID from two sources (unlikely but possible)
        mock_reddit.return_value = [_make_item("same_id")]
        mock_yt.return_value = [_make_item("same_id", "youtube")]
        mock_gn.return_value = []

        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(sources=["reddit", "youtube", "google_news"], db_path=tmp_path / "test.db")

        assert summary["total_unique"] == 1

    @patch("src.scrapers.reddit_scraper.scrape_reddit")
    @patch("src.scrapers.youtube_scraper.scrape_youtube")
    @patch("src.scrapers.google_rss_scraper.scrape_google_trends")
    def test_continues_if_one_source_fails(self, mock_gn, mock_yt, mock_reddit, tmp_path):
        mock_reddit.side_effect = Exception("Reddit is down")
        mock_yt.return_value = [_make_item("y1", "youtube")]
        mock_gn.return_value = [_make_item("g1", "google_news")]

        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(sources=["reddit", "youtube", "google_news"], db_path=tmp_path / "test.db")

        assert summary["sources"]["reddit"]["status"] == "error"
        assert summary["sources"]["youtube"]["status"] == "ok"
        assert summary["total_inserted"] == 2  # YouTube + Google News still worked

    @patch("src.scrapers.reddit_scraper.scrape_reddit")
    def test_runs_single_source(self, mock_reddit, tmp_path):
        mock_reddit.return_value = [_make_item("r1")]

        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(sources=["reddit"], db_path=tmp_path / "test.db")

        assert "reddit" in summary["sources"]
        assert "youtube" not in summary["sources"]

    @patch("src.scrapers.reddit_scraper.scrape_reddit")
    def test_second_run_skips_existing(self, mock_reddit, tmp_path):
        items = [_make_item("r1"), _make_item("r2")]
        mock_reddit.return_value = items

        from src.scrapers.run_all import run_pipeline
        db = tmp_path / "test.db"

        run_pipeline(sources=["reddit"], db_path=db)
        summary2 = run_pipeline(sources=["reddit"], db_path=db)

        assert summary2["total_inserted"] == 0
        assert summary2["total_skipped"] == 2
