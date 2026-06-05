"""Tests for the pipeline runner (run_all.py).

Covers the two-track collection logic: verified articles (RSS/NewsAPI)
are collected and linked per topic. All scraper and NewsAPI HTTP calls are mocked.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.scrapers.run_all import (
    _extract_candidates,
    _generate_query_variants,
    _pool_matches,
)
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_item(
    id: str,
    title: str = "Bundesrat lehnt Antrag ab",
    platform: str = "newsapi",
    url: str | None = None,
) -> RawItem:
    return {
        "id": id,
        "title": title,
        "description": None,
        "source": "test",
        "url": url or f"https://reuters.com/{id}",
        "platform": platform,
        "timestamp": "2026-06-01T10:00:00Z",
        "engagement": {"score": 0, "comments": 0},
    }


# ---------------------------------------------------------------------------
# _extract_candidates
# ---------------------------------------------------------------------------

class TestExtractCandidates:
    def test_returns_proper_nouns_and_all_candidates(self):
        proper, candidates = _extract_candidates("Bundesrat lehnt Antrag ab")
        assert "Bundesrat" in proper
        assert "Antrag" in proper

    def test_stopwords_excluded(self):
        _, candidates = _extract_candidates("der die das und oder")
        assert candidates == []

    def test_proper_subset_of_candidates(self):
        proper, candidates = _extract_candidates("Merz besucht Berlin")
        assert set(proper) <= set(candidates)


# ---------------------------------------------------------------------------
# _pool_matches
# ---------------------------------------------------------------------------

class TestPoolMatches:
    def test_matching_keyword_returns_true(self):
        item = _make_item("a1", title="Bundesrat stimmt zu")
        assert _pool_matches(item, ["bundesrat"]) is True

    def test_no_match_returns_false(self):
        item = _make_item("a2", title="Wetter in Berlin")
        assert _pool_matches(item, ["migration", "merz"]) is False

    def test_empty_candidates_returns_false(self):
        item = _make_item("a3")
        assert _pool_matches(item, []) is False


# ---------------------------------------------------------------------------
# _generate_query_variants
# ---------------------------------------------------------------------------

class TestGenerateQueryVariants:
    def test_returns_list(self):
        variants = _generate_query_variants("Bundesrat lehnt Migration ab")
        assert isinstance(variants, list)
        assert len(variants) > 0

    def test_no_duplicates(self):
        variants = _generate_query_variants("Merz besucht Berlin CDU")
        assert len(variants) == len(set(variants))

    def test_proper_noun_pairs_come_first(self):
        variants = _generate_query_variants("Merz Berlin CDU")
        # First variant should be a pair of proper nouns
        assert len(variants[0].split()) == 2


# ---------------------------------------------------------------------------
# run_pipeline (mocked scrapers)
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @patch("src.scrapers.run_all.scrape_newsapi")
    @patch("src.scrapers.run_all.scrape_rss_sources")
    @patch("src.scrapers.run_all.scrape_google_trends")
    def test_pipeline_returns_summary_dict(
        self, mock_gn, mock_rss, mock_newsapi, tmp_path
    ):
        mock_gn.return_value = [_make_item("seed1", "Bundesrat Migration Antrag", "google_news")]
        mock_rss.return_value = []
        # Return enough verified articles to qualify the topic
        mock_newsapi.return_value = [
            _make_item(f"n{i}", "Bundesrat Migrationspolitik") for i in range(20)
        ]

        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(
            target_topics=1,
            articles_per_topic=20,
            db_path=tmp_path / "test.db",
            skip_newsapi=False,
        )
        assert "topics_created" in summary
        assert "articles_inserted" in summary

    @patch("src.scrapers.run_all.scrape_newsapi")
    @patch("src.scrapers.run_all.scrape_rss_sources")
    @patch("src.scrapers.run_all.scrape_google_trends")
    def test_topic_drops_when_insufficient_verified_articles(
        self, mock_gn, mock_rss, mock_newsapi, tmp_path
    ):
        mock_gn.return_value = [_make_item("seed1", "Bundesrat Migration", "google_news")]
        mock_rss.return_value = []
        # Only 5 verified articles — below the 20 threshold
        mock_newsapi.return_value = [
            _make_item(f"n{i}", "Bundesrat Migrationspolitik") for i in range(5)
        ]

        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(
            target_topics=1,
            articles_per_topic=20,
            db_path=tmp_path / "test.db",
        )
        assert summary["topics_created"] == 0
        assert summary["topics_dropped"] >= 1

    @patch("src.scrapers.run_all.scrape_google_trends")
    def test_no_rss_seeds_returns_early(self, mock_gn, tmp_path):
        mock_gn.return_value = []
        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(db_path=tmp_path / "test.db")
        assert summary["rss_seeds"] == 0
        assert summary["topics_created"] == 0
