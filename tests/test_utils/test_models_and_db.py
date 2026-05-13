"""Tests for src/utils/models.py and src/utils/db.py.

Verifies the RawItem contract and all database CRUD operations
including deduplication, filtering, and JSON round-tripping.
"""

from __future__ import annotations

import json
import pytest

from src.utils.db import init_db, insert_items, get_items
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_item(
    id: str = "reddit_abc123",
    title: str = "Test headline",
    platform: str = "reddit",
    **overrides,
) -> RawItem:
    """Factory for creating RawItem dicts with sensible defaults."""
    base: RawItem = {
        "id": id,
        "title": title,
        "description": "Some body text",
        "source": "r/worldnews",
        "url": "https://reddit.com/r/worldnews/abc123",
        "platform": platform,
        "timestamp": "2026-05-11T10:00:00Z",
        "engagement": {"score": 150, "comments": 42},
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


@pytest.fixture
def db_conn(tmp_path):
    """Provide a fresh in-memory-like DB connection per test."""
    db_path = tmp_path / "test.db"
    conn = init_db(db_path)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# RawItem model tests
# ---------------------------------------------------------------------------

class TestRawItem:
    """Verify the RawItem TypedDict matches the Issue #1 contract."""

    def test_all_required_fields_present(self):
        item = _make_item()
        required_keys = {
            "id", "title", "description", "source",
            "url", "platform", "timestamp", "engagement",
        }
        assert set(item.keys()) == required_keys

    def test_engagement_has_score_and_comments(self):
        item = _make_item()
        assert "score" in item["engagement"]
        assert "comments" in item["engagement"]

    def test_description_can_be_none(self):
        item = _make_item(description=None)
        assert item["description"] is None

    def test_platform_values(self):
        for platform in ("reddit", "youtube", "google_news", "newsapi"):
            item = _make_item(platform=platform)
            assert item["platform"] == platform


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase:
    """Verify init, insert, dedup, and read operations."""

    def test_init_creates_table(self, db_conn):
        tables = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "raw_items" in table_names

    def test_insert_single_item(self, db_conn):
        item = _make_item()
        inserted = insert_items(db_conn, [item])
        assert inserted == 1

    def test_insert_multiple_items(self, db_conn):
        items = [
            _make_item(id="reddit_1"),
            _make_item(id="reddit_2"),
            _make_item(id="youtube_1", platform="youtube"),
        ]
        inserted = insert_items(db_conn, items)
        assert inserted == 3

    def test_dedup_skips_existing_ids(self, db_conn):
        item = _make_item(id="reddit_dup")
        insert_items(db_conn, [item])
        inserted_again = insert_items(db_conn, [item])
        assert inserted_again == 0

    def test_dedup_mixed_new_and_existing(self, db_conn):
        item1 = _make_item(id="reddit_old")
        insert_items(db_conn, [item1])

        item2 = _make_item(id="reddit_new")
        inserted = insert_items(db_conn, [item1, item2])
        assert inserted == 1

    def test_get_items_returns_all(self, db_conn):
        items = [_make_item(id=f"item_{i}") for i in range(5)]
        insert_items(db_conn, items)
        results = get_items(db_conn)
        assert len(results) == 5

    def test_get_items_filter_by_platform(self, db_conn):
        items = [
            _make_item(id="r1", platform="reddit"),
            _make_item(id="y1", platform="youtube"),
            _make_item(id="g1", platform="google_news"),
        ]
        insert_items(db_conn, items)
        reddit_only = get_items(db_conn, platform="reddit")
        assert len(reddit_only) == 1
        assert reddit_only[0]["platform"] == "reddit"

    def test_get_items_filter_by_since(self, db_conn):
        items = [
            _make_item(id="old", timestamp="2026-05-01T00:00:00Z"),
            _make_item(id="new", timestamp="2026-05-11T12:00:00Z"),
        ]
        insert_items(db_conn, items)
        recent = get_items(db_conn, since="2026-05-10T00:00:00Z")
        assert len(recent) == 1
        assert recent[0]["id"] == "new"

    def test_get_items_with_limit(self, db_conn):
        items = [_make_item(id=f"item_{i}") for i in range(10)]
        insert_items(db_conn, items)
        limited = get_items(db_conn, limit=3)
        assert len(limited) == 3

    def test_engagement_json_round_trips(self, db_conn):
        engagement = {"score": 999, "comments": 77}
        item = _make_item(engagement=engagement)
        insert_items(db_conn, [item])
        result = get_items(db_conn)[0]
        assert result["engagement"] == engagement
        assert isinstance(result["engagement"]["score"], int)

    def test_nlp_columns_are_null_initially(self, db_conn):
        insert_items(db_conn, [_make_item()])
        result = get_items(db_conn)[0]
        assert result["cleaned_text"] is None
        assert result["tokens"] is None
        assert result["lemmas"] is None
        assert result["entities"] is None
        assert result["keywords"] is None
