"""Tests for the pipeline runner (run_all.py).

Covers the two-track collection logic: verified articles (RSS/NewsAPI)
and Reddit posts are collected into separate pools and linked independently.
All scraper and NewsAPI HTTP calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.scrapers.run_all import (
    _extract_candidates,
    _fetch_reddit_for_topic,
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


def _make_reddit_item(id: str, title: str = "Bundesrat Diskussion") -> RawItem:
    return _make_item(
        id=id,
        title=title,
        platform="reddit",
        url=f"https://www.reddit.com/r/de/comments/{id}/",
    )


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
# _fetch_reddit_for_topic
# ---------------------------------------------------------------------------

class TestFetchRedditForTopic:
    def test_returns_matching_posts(self):
        pool = {
            "r1": _make_reddit_item("r1", "Bundesrat debattiert Migrationspolitik"),
            "r2": _make_reddit_item("r2", "Guten Morgen aus Berlin"),
        }
        result = _fetch_reddit_for_topic("Bundesrat Migrationspolitik", pool)
        ids = [r["id"] for r in result]
        assert "r1" in ids
        assert "r2" not in ids

    def test_respects_max_count(self):
        pool = {
            f"r{i}": _make_reddit_item(f"r{i}", "Bundesrat Diskussion heute")
            for i in range(10)
        }
        result = _fetch_reddit_for_topic("Bundesrat", pool, max_count=3)
        assert len(result) <= 3

    def test_empty_pool_returns_empty(self):
        result = _fetch_reddit_for_topic("Bundesrat Migration", {})
        assert result == []

    def test_no_match_returns_empty(self):
        pool = {"r1": _make_reddit_item("r1", "Fußball Ergebnisse")}
        result = _fetch_reddit_for_topic("Bundesrat Migration", pool)
        assert result == []

    def test_all_platforms_in_result_are_reddit(self):
        pool = {
            "r1": _make_reddit_item("r1", "Bundesrat diskutiert"),
            "r2": _make_reddit_item("r2", "Bundesrat heute"),
        }
        result = _fetch_reddit_for_topic("Bundesrat", pool)
        assert all(r["platform"] == "reddit" for r in result)


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
    @patch("src.scrapers.run_all.scrape_reddit_by_keywords")
    @patch("src.scrapers.run_all.scrape_reddit_german")
    @patch("src.scrapers.run_all.scrape_rss_sources")
    @patch("src.scrapers.run_all.scrape_google_trends")
    def test_pipeline_returns_summary_dict(
        self, mock_gn, mock_rss, mock_reddit_de, mock_reddit_kw, mock_newsapi, tmp_path
    ):
        mock_gn.return_value = [_make_item("seed1", "Bundesrat Migration Antrag", "google_news")]
        mock_rss.return_value = []
        mock_reddit_de.return_value = []
        mock_reddit_kw.return_value = []
        # Return enough verified articles to qualify the topic
        mock_newsapi.return_value = [
            _make_item(f"n{i}", "Bundesrat Migrationspolitik") for i in range(20)
        ]

        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(
            target_topics=1,
            articles_per_topic=20,
            reddit_per_topic=0,
            db_path=tmp_path / "test.db",
            skip_newsapi=False,
        )
        assert "topics_created" in summary
        assert "articles_inserted" in summary

    @patch("src.scrapers.run_all.scrape_newsapi")
    @patch("src.scrapers.run_all.scrape_reddit_by_keywords")
    @patch("src.scrapers.run_all.scrape_reddit_german")
    @patch("src.scrapers.run_all.scrape_rss_sources")
    @patch("src.scrapers.run_all.scrape_google_trends")
    def test_topic_drops_when_insufficient_verified_articles(
        self, mock_gn, mock_rss, mock_reddit_de, mock_reddit_kw, mock_newsapi, tmp_path
    ):
        mock_gn.return_value = [_make_item("seed1", "Bundesrat Migration", "google_news")]
        mock_rss.return_value = []
        mock_reddit_de.return_value = [_make_reddit_item("r1", "Bundesrat Migration")]
        mock_reddit_kw.return_value = []
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

    @patch("src.scrapers.run_all.scrape_newsapi")
    @patch("src.scrapers.run_all.scrape_reddit_by_keywords")
    @patch("src.scrapers.run_all.scrape_reddit_german")
    @patch("src.scrapers.run_all.scrape_rss_sources")
    @patch("src.scrapers.run_all.scrape_google_trends")
    def test_reddit_articles_linked_separately_from_verified(
        self, mock_gn, mock_rss, mock_reddit_de, mock_reddit_kw, mock_newsapi, tmp_path
    ):
        """Both verified and Reddit articles should be in topic_sources."""
        mock_gn.return_value = [_make_item("seed1", "Bundesrat Migration", "google_news")]
        mock_rss.return_value = []
        mock_reddit_de.return_value = [
            _make_reddit_item("reddit1", "Bundesrat diskutiert Migration"),
        ]
        mock_reddit_kw.return_value = []
        mock_newsapi.return_value = [
            _make_item(f"n{i}", "Bundesrat Migrationspolitik") for i in range(20)
        ]

        from src.scrapers.run_all import run_pipeline
        from src.utils.db import init_db
        db = tmp_path / "test.db"
        run_pipeline(
            target_topics=1,
            articles_per_topic=20,
            reddit_per_topic=5,
            db_path=db,
        )
        conn = init_db(db)
        platforms = {
            r["platform"]
            for r in conn.execute(
                "SELECT ri.platform FROM topic_sources ts "
                "JOIN raw_items ri ON ri.id = ts.item_id"
            ).fetchall()
        }
        conn.close()
        assert "newsapi" in platforms
        assert "reddit" in platforms

    @patch("src.scrapers.run_all.scrape_google_trends")
    def test_no_rss_seeds_returns_early(self, mock_gn, tmp_path):
        mock_gn.return_value = []
        from src.scrapers.run_all import run_pipeline
        summary = run_pipeline(db_path=tmp_path / "test.db")
        assert summary["rss_seeds"] == 0
        assert summary["topics_created"] == 0
