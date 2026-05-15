"""Tests for src/utils/clustering.py.

Covers title cleaning, fuzzy merging, DB rebuild, and the full cluster_items
pipeline. All tests use an in-memory SQLite DB — no external services touched.
"""

from __future__ import annotations

import pytest

from src.utils.clustering import _clean_title, _fuzzy_merge, cluster_items
from src.utils.db import init_db, insert_items
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_item(id: str, title: str, platform: str = "reddit") -> RawItem:
    return {
        "id": id,
        "title": title,
        "description": None,
        "source": "r/news",
        "url": f"https://reddit.com/{id}",
        "platform": platform,
        "timestamp": "2026-05-15T10:00:00Z",
        "engagement": {"score": 10, "comments": 2},
    }


@pytest.fixture
def db_conn(tmp_path):
    conn = init_db(tmp_path / "test.db")
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# _clean_title
# ---------------------------------------------------------------------------

class TestCleanTitle:
    def test_lowercases(self):
        assert _clean_title("BREAKING NEWS") == "breaking news"

    def test_strips_punctuation(self):
        assert _clean_title("Hello, world!") == "hello world"

    def test_collapses_whitespace(self):
        assert _clean_title("too  many   spaces") == "too many spaces"

    def test_strips_leading_trailing(self):
        assert _clean_title("  leading ") == "leading"

    def test_empty_string(self):
        assert _clean_title("") == ""


# ---------------------------------------------------------------------------
# _fuzzy_merge
# ---------------------------------------------------------------------------

class TestFuzzyMerge:
    def test_merges_near_duplicate_singletons(self):
        # Two very similar titles that should merge
        labels = [0, 1]
        cleaned = ["covid vaccine update", "covid vaccines update"]
        result = _fuzzy_merge(labels, cleaned, threshold=75)
        assert result[0] == result[1]

    def test_does_not_merge_dissimilar_titles(self):
        labels = [0, 1]
        cleaned = ["stock market crash", "local weather forecast"]
        result = _fuzzy_merge(labels, cleaned, threshold=75)
        assert result[0] != result[1]

    def test_singleton_merges_into_larger_cluster(self):
        # labels: 0,0,0 form a cluster; 1 is singleton similar to them
        labels = [0, 0, 0, 1]
        cleaned = [
            "bitcoin price rise",
            "bitcoin prices rise",
            "bitcoin price rises",
            "bitcoin price rising",
        ]
        result = _fuzzy_merge(labels, cleaned, threshold=75)
        # The singleton at index 3 should join cluster 0
        assert result[3] == 0

    def test_no_change_when_no_singletons(self):
        labels = [0, 0, 1, 1]
        cleaned = ["a b c", "a b d", "x y z", "x y w"]
        result = _fuzzy_merge(labels[:], cleaned, threshold=75)
        # No singletons — labels unchanged
        assert Counter(result) == Counter(labels)

    def test_returns_list(self):
        result = _fuzzy_merge([0, 1], ["foo bar", "baz qux"], threshold=75)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# cluster_items — integration (uses real DB)
# ---------------------------------------------------------------------------

class TestClusterItems:
    def test_empty_db_returns_zero(self, db_conn):
        n = cluster_items(db_conn)
        assert n == 0

    def test_single_item_creates_one_topic(self, db_conn):
        insert_items(db_conn, [_make_item("r_1", "Breaking news headline")])
        n = cluster_items(db_conn)
        assert n == 1

    def test_similar_titles_cluster_together(self, db_conn):
        items = [
            _make_item("r_1", "Scientists discover new vaccine for COVID-19"),
            _make_item("r_2", "Scientists discover COVID-19 vaccine breakthrough"),
            _make_item("r_3", "New COVID vaccine discovered by scientists"),
        ]
        insert_items(db_conn, items)
        n = cluster_items(db_conn, distance_threshold=0.35)
        # All three should land in a single cluster
        assert n == 1

    def test_dissimilar_titles_produce_multiple_topics(self, db_conn):
        items = [
            _make_item("r_1", "Bitcoin price surges to new record high"),
            _make_item("r_2", "Heavy rainfall causes flooding in Texas"),
            _make_item("r_3", "Olympics opening ceremony draws global audience"),
        ]
        insert_items(db_conn, items)
        n = cluster_items(db_conn, distance_threshold=0.35)
        assert n == 3

    def test_cluster_id_written_to_raw_items(self, db_conn):
        items = [
            _make_item("r_1", "US election results announced nationwide"),
            _make_item("r_2", "US election results announced across nation"),
        ]
        insert_items(db_conn, items)
        cluster_items(db_conn)

        rows = db_conn.execute(
            "SELECT id, cluster_id FROM raw_items ORDER BY id"
        ).fetchall()
        cluster_ids = [r["cluster_id"] for r in rows]
        assert all(cid is not None for cid in cluster_ids)
        assert cluster_ids[0] == cluster_ids[1]

    def test_topics_table_populated(self, db_conn):
        insert_items(db_conn, [_make_item("r_1", "Wildfire spreads in California")])
        cluster_items(db_conn)

        topics = db_conn.execute("SELECT * FROM topics").fetchall()
        assert len(topics) == 1
        assert topics[0]["label"] == "Wildfire spreads in California"
        assert topics[0]["item_count"] == 1

    def test_topic_sources_table_populated(self, db_conn):
        items = [
            _make_item("r_1", "Tech giant announces layoffs amid recession fears"),
            _make_item("r_2", "Tech company announces mass layoffs"),
        ]
        insert_items(db_conn, items)
        cluster_items(db_conn)

        sources = db_conn.execute("SELECT * FROM topic_sources").fetchall()
        assert len(sources) == 2

    def test_rerun_replaces_previous_topics(self, db_conn):
        insert_items(db_conn, [_make_item("r_1", "First run headline")])
        cluster_items(db_conn)

        insert_items(db_conn, [_make_item("r_2", "Second run headline")])
        cluster_items(db_conn)

        topic_count = db_conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        source_count = db_conn.execute(
            "SELECT COUNT(*) FROM topic_sources"
        ).fetchone()[0]
        assert topic_count >= 1
        assert source_count == 2  # both items covered

    def test_topics_and_raw_items_cluster_ids_consistent(self, db_conn):
        items = [_make_item(f"r_{i}", f"Unique story number {i}") for i in range(5)]
        insert_items(db_conn, items)
        cluster_items(db_conn)

        topic_ids = {
            r["id"]
            for r in db_conn.execute("SELECT id FROM topics").fetchall()
        }
        raw_cluster_ids = {
            r["cluster_id"]
            for r in db_conn.execute("SELECT cluster_id FROM raw_items").fetchall()
        }
        assert raw_cluster_ids == topic_ids


# Needed for test_no_change_when_no_singletons
from collections import Counter
