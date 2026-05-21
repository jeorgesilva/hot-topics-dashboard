"""Tests for src/scoring/source_trust.py.

All tests use in-process fixtures — no real CSV file or network calls.
The trust DB is patched at the module level where needed.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pytest

from src.scoring.source_trust import (
    HIGH_TRUST_THRESHOLD,
    NEUTRAL_SCORE,
    _UNKNOWN_BREAKING_SCORE,
    _UNKNOWN_SCORE,
    _domain_from_url,
    _load_trust_db,
    compute_coverage_metrics,
    get_trust_score,
    score_coverage,
)
from src.utils.db import init_db, insert_items
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = """\
domain,trust_score,factual_reporting,bias
reuters.com,94,VERY HIGH,CENTER
bbc.com,86,HIGH,LEFT-CENTER
foxnews.com,38,MIXED,RIGHT
infowars.com,2,VERY LOW,RIGHT
"""

_SAMPLE_DB = {
    "reuters.com": 94.0,
    "bbc.com": 86.0,
    "foxnews.com": 38.0,
    "infowars.com": 2.0,
}


def _make_item(id: str, url: str, platform: str = "newsapi") -> RawItem:
    return {
        "id": id,
        "title": f"Article {id}",
        "description": None,
        "source": "test",
        "url": url,
        "platform": platform,
        "timestamp": "2026-05-19T10:00:00Z",
        "engagement": {"score": 0, "comments": 0},
    }


@pytest.fixture
def db_conn(tmp_path):
    conn = init_db(tmp_path / "test.db")
    yield conn
    conn.close()


@pytest.fixture
def seeded_db(db_conn):
    """DB with one topic (id=0) and three articles from known domains."""
    items = [
        _make_item("a1", "https://reuters.com/article/1"),
        _make_item("a2", "https://bbc.com/news/2"),
        _make_item("a3", "https://infowars.com/story/3"),
    ]
    insert_items(db_conn, items)
    with db_conn:
        db_conn.execute(
            "INSERT INTO topics (id, label, created_at, item_count) VALUES (0, 'Test Topic', '2026-05-19T10:00:00Z', 3)"
        )
        db_conn.executemany(
            "INSERT INTO topic_sources (topic_id, item_id) VALUES (0, ?)",
            [("a1",), ("a2",), ("a3",)],
        )
    return db_conn


# ---------------------------------------------------------------------------
# _load_trust_db
# ---------------------------------------------------------------------------

class TestLoadTrustDb:
    def test_parses_known_domains(self, tmp_path):
        csv_path = tmp_path / "trust.csv"
        csv_path.write_text(_SAMPLE_CSV)
        db = _load_trust_db(csv_path)
        assert db["reuters.com"] == 94.0
        assert db["bbc.com"] == 86.0

    def test_strips_www_prefix(self, tmp_path):
        csv_path = tmp_path / "trust.csv"
        csv_path.write_text("domain,trust_score,factual_reporting,bias\nwww.bbc.com,86,HIGH,CENTER\n")
        db = _load_trust_db(csv_path)
        assert "bbc.com" in db
        assert "www.bbc.com" not in db

    def test_missing_file_returns_empty_dict(self, tmp_path):
        db = _load_trust_db(tmp_path / "nonexistent.csv")
        assert db == {}

    def test_skips_non_numeric_score(self, tmp_path):
        csv_path = tmp_path / "trust.csv"
        csv_path.write_text("domain,trust_score,factual_reporting,bias\nbad.com,NOT_A_NUMBER,HIGH,CENTER\n")
        db = _load_trust_db(csv_path)
        assert "bad.com" not in db


# ---------------------------------------------------------------------------
# get_trust_score
# ---------------------------------------------------------------------------

class TestGetTrustScore:
    def test_known_domain_returns_score(self):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            assert get_trust_score("reuters.com") == 94.0

    def test_www_prefix_stripped(self):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            assert get_trust_score("www.bbc.com") == 86.0

    def test_unknown_domain_returns_lower_default(self):
        # Unknown domains are penalised below 50 — a missing domain is more
        # likely to be a shell site than a genuinely neutral outlet.
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            assert get_trust_score("unknown-site.com") == _UNKNOWN_SCORE

    def test_unknown_domain_breaking_topic_returns_lower_score(self):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            score = get_trust_score("unknown-site.com", topic_is_breaking=True)
            assert score == _UNKNOWN_BREAKING_SCORE
            assert score < _UNKNOWN_SCORE

    def test_explicit_neutral_overrides_contextual_default(self):
        # Callers that manage their own neutral (e.g. compute_coverage_metrics)
        # can still pass it explicitly and it takes precedence.
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            assert get_trust_score("unknown-site.com", neutral=30.0) == 30.0

    def test_known_domain_unaffected_by_topic_is_breaking(self):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            assert get_trust_score("reuters.com", topic_is_breaking=True) == 94.0

    def test_case_insensitive(self):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            assert get_trust_score("BBC.COM") == 86.0


# ---------------------------------------------------------------------------
# _domain_from_url
# ---------------------------------------------------------------------------

class TestDomainFromUrl:
    def test_extracts_domain(self):
        assert _domain_from_url("https://www.bbc.com/news/article") == "bbc.com"

    def test_strips_www(self):
        assert _domain_from_url("https://www.reuters.com/article") == "reuters.com"

    def test_no_scheme(self):
        result = _domain_from_url("bbc.com/news")
        assert "bbc.com" in result


# ---------------------------------------------------------------------------
# compute_coverage_metrics
# ---------------------------------------------------------------------------

class TestComputeCoverageMetrics:
    def test_avg_trust_correct(self, seeded_db):
        # reuters=94, bbc=86, infowars=2 → avg = (94+86+2)/3 = 60.667
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            m = compute_coverage_metrics(0, seeded_db)
        assert abs(m["avg_trust"] - (94 + 86 + 2) / 3) < 0.01

    def test_trust_variance_nonzero_for_mixed_sources(self, seeded_db):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            m = compute_coverage_metrics(0, seeded_db)
        assert m["trust_variance"] > 0.0

    def test_coverage_breadth_counts_credible_domains(self, seeded_db):
        # reuters(94) and bbc(86) are above HIGH_TRUST_THRESHOLD=60; infowars(2) is not
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            m = compute_coverage_metrics(0, seeded_db)
        assert m["coverage_breadth"] == 2

    def test_coverage_ratio_correct(self, seeded_db):
        # 2 of 3 articles are from credible sources
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            m = compute_coverage_metrics(0, seeded_db)
        assert abs(m["coverage_ratio"] - 2 / 3) < 0.01

    def test_empty_topic_returns_neutral_defaults(self, db_conn):
        with db_conn:
            db_conn.execute(
                "INSERT INTO topics (id, label, created_at, item_count) VALUES (99, 'Empty', '2026-05-19T10:00:00Z', 0)"
            )
        m = compute_coverage_metrics(99, db_conn)
        assert m["avg_trust"] == NEUTRAL_SCORE
        assert m["trust_variance"] == 0.0
        assert m["coverage_breadth"] == 0
        assert m["coverage_ratio"] == 0.0

    def test_all_credible_sources_ratio_is_one(self, db_conn):
        items = [
            _make_item("b1", "https://reuters.com/a"),
            _make_item("b2", "https://bbc.com/b"),
        ]
        insert_items(db_conn, items)
        with db_conn:
            db_conn.execute(
                "INSERT INTO topics (id, label, created_at, item_count) VALUES (1, 'T', '2026-05-19T10:00:00Z', 2)"
            )
            db_conn.executemany(
                "INSERT INTO topic_sources (topic_id, item_id) VALUES (1, ?)",
                [("b1",), ("b2",)],
            )
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            m = compute_coverage_metrics(1, db_conn)
        assert m["coverage_ratio"] == 1.0
        assert m["trust_variance"] == pytest.approx(
            math.sqrt(((94 - 90) ** 2 + (86 - 90) ** 2) / 2), abs=0.01
        )


# ---------------------------------------------------------------------------
# score_coverage (integration)
# ---------------------------------------------------------------------------

class TestScoreCoverage:
    def test_upserts_rows_into_topic_scores(self, seeded_db):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            count = score_coverage(seeded_db)
        assert count == 1
        row = seeded_db.execute(
            "SELECT avg_trust FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row is not None
        assert row["avg_trust"] is not None

    def test_idempotent_on_rerun(self, seeded_db):
        with patch("src.scoring.source_trust._TRUST_DB", _SAMPLE_DB):
            score_coverage(seeded_db)
            count2 = score_coverage(seeded_db)
        assert count2 == 1
        assert seeded_db.execute("SELECT COUNT(*) FROM topic_scores").fetchone()[0] == 1

    def test_empty_topics_returns_zero(self, db_conn):
        count = score_coverage(db_conn)
        assert count == 0
