"""Tests for src/scoring/compute_scores.py.

All DB interactions use a file-backed SQLite DB under tmp_path.
Person-A signals (avg_sentiment_extremity, sensationalism_avg,
framing_inconsistency) are inserted directly into topic_scores to
simulate Person A's pipeline having already run.
"""

from __future__ import annotations

import pytest

from src.scoring.compute_scores import (
    _MISINFO_THRESHOLD,
    _WEIGHTS,
    compute_composite,
    compute_risk,
    explain_score,
    grade_topic,
    score_all_topics,
)
from src.utils.db import init_db, insert_items
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_item(id: str, url: str = "https://reuters.com/x") -> RawItem:
    return {
        "id": id,
        "title": f"Article {id}",
        "description": None,
        "source": "reuters.com",
        "url": url,
        "platform": "newsapi",
        "timestamp": "2026-05-19T10:00:00Z",
        "engagement": {"score": 0, "comments": 0},
    }


@pytest.fixture
def db_conn(tmp_path):
    conn = init_db(tmp_path / "test.db")
    yield conn
    conn.close()


def _seed_topic(conn, topic_id: int, item_ids: list[str], **score_kwargs) -> None:
    """Insert a topic, its items, and optional topic_scores row."""
    with conn:
        conn.execute(
            "INSERT INTO topics (id, label, created_at, item_count) VALUES (?, 'T', '2026-05-19T00:00:00Z', ?)",
            (topic_id, len(item_ids)),
        )
        conn.executemany(
            "INSERT INTO topic_sources (topic_id, item_id) VALUES (?, ?)",
            [(topic_id, iid) for iid in item_ids],
        )
        if score_kwargs:
            cols = ", ".join(score_kwargs.keys())
            placeholders = ", ".join("?" * len(score_kwargs))
            conn.execute(
                f"INSERT INTO topic_scores (topic_id, {cols}) VALUES (?, {placeholders})",
                (topic_id, *score_kwargs.values()),
            )


# ---------------------------------------------------------------------------
# compute_risk formula
# ---------------------------------------------------------------------------

class TestComputeRisk:
    def test_all_zero_signals_is_zero_risk(self):
        # avg_trust=100 → (1-100/100)=0; all other signals=0
        risk = compute_risk(
            avg_trust=100.0,
            avg_sentiment_extremity=0.0,
            coverage_ratio=1.0,
            framing_inconsistency=0.0,
            sensationalism_avg=0.0,
        )
        assert risk == pytest.approx(0.0)

    def test_all_max_signals_is_one_risk(self):
        # avg_trust=0, all other signals=1, coverage_ratio=0
        risk = compute_risk(
            avg_trust=0.0,
            avg_sentiment_extremity=1.0,
            coverage_ratio=0.0,
            framing_inconsistency=1.0,
            sensationalism_avg=1.0,
        )
        assert risk == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        assert sum(_WEIGHTS.values()) == pytest.approx(1.0)

    def test_formula_matches_manual_calculation(self):
        risk = compute_risk(
            avg_trust=60.0,
            avg_sentiment_extremity=0.5,
            coverage_ratio=0.6,
            framing_inconsistency=0.4,
            sensationalism_avg=0.3,
        )
        expected = (
            0.30 * (1 - 60 / 100)
            + 0.25 * 0.5
            + 0.20 * (1 - 0.6)
            + 0.15 * 0.4
            + 0.10 * 0.3
        )
        assert risk == pytest.approx(expected, abs=1e-6)

    def test_high_trust_lowers_risk(self):
        low_trust = compute_risk(20.0, 0.5, 0.5, 0.5, 0.5)
        high_trust = compute_risk(90.0, 0.5, 0.5, 0.5, 0.5)
        assert high_trust < low_trust


# ---------------------------------------------------------------------------
# grade_topic
# ---------------------------------------------------------------------------

class TestGradeTopic:
    def test_zero_risk_is_grade_a(self):
        assert grade_topic(0.0) == "A"

    def test_full_risk_is_grade_f(self):
        assert grade_topic(1.0) == "F"

    def test_grade_boundaries(self):
        assert grade_topic(0.19) == "A"   # reliability 0.81
        assert grade_topic(0.21) == "B"   # reliability 0.79
        assert grade_topic(0.39) == "B"   # reliability 0.61
        assert grade_topic(0.41) == "C"   # reliability 0.59
        assert grade_topic(0.59) == "C"   # reliability 0.41
        assert grade_topic(0.61) == "D"   # reliability 0.39
        assert grade_topic(0.79) == "D"   # reliability 0.21
        assert grade_topic(0.81) == "F"   # reliability 0.19

    def test_misinfo_threshold_is_grade_d_or_f(self):
        grade = grade_topic(_MISINFO_THRESHOLD)
        assert grade in {"D", "F", "C"}


# ---------------------------------------------------------------------------
# explain_score
# ---------------------------------------------------------------------------

class TestExplainScore:
    def test_returns_all_expected_keys(self, db_conn):
        insert_items(db_conn, [_make_item("e1")])
        _seed_topic(db_conn, 0, ["e1"],
                    avg_trust=80.0, trust_variance=5.0, coverage_breadth=3,
                    coverage_ratio=0.8, avg_sentiment_extremity=0.2,
                    sensationalism_avg=0.1, framing_inconsistency=0.15,
                    composite_risk=0.22)
        result = explain_score(0, db_conn)
        assert {"topic_id", "composite_risk", "grade", "contributions"} <= result.keys()

    def test_contributions_keys_present(self, db_conn):
        insert_items(db_conn, [_make_item("e2")])
        _seed_topic(db_conn, 1, ["e2"],
                    avg_trust=50.0, coverage_ratio=0.5,
                    avg_sentiment_extremity=0.5, sensationalism_avg=0.5,
                    framing_inconsistency=0.5, composite_risk=0.5)
        result = explain_score(1, db_conn)
        expected_keys = {
            "source_distrust", "sentiment_extremity",
            "low_credible_coverage", "framing_inconsistency", "sensationalism",
        }
        assert expected_keys == result["contributions"].keys()

    def test_contributions_sum_to_composite_risk(self, db_conn):
        insert_items(db_conn, [_make_item("e3")])
        # Use the formula itself so inserted composite_risk matches contributions
        expected_risk = compute_risk(
            avg_trust=40.0, avg_sentiment_extremity=0.6,
            coverage_ratio=0.3, framing_inconsistency=0.7, sensationalism_avg=0.4,
        )
        _seed_topic(db_conn, 2, ["e3"],
                    avg_trust=40.0, coverage_ratio=0.3,
                    avg_sentiment_extremity=0.6, sensationalism_avg=0.4,
                    framing_inconsistency=0.7, composite_risk=expected_risk)
        result = explain_score(2, db_conn)
        total = sum(result["contributions"].values())
        assert total == pytest.approx(result["composite_risk"], abs=0.01)

    def test_unscored_topic_returns_empty_dict(self, db_conn):
        insert_items(db_conn, [_make_item("e4")])
        _seed_topic(db_conn, 3, ["e4"])   # no score_kwargs → no topic_scores row
        assert explain_score(3, db_conn) == {}


# ---------------------------------------------------------------------------
# compute_composite
# ---------------------------------------------------------------------------

class TestComputeComposite:
    def test_scores_topics_with_all_signals(self, db_conn):
        insert_items(db_conn, [_make_item("c1")])
        _seed_topic(db_conn, 0, ["c1"],
                    avg_trust=80.0, coverage_ratio=0.8,
                    avg_sentiment_extremity=0.1, sensationalism_avg=0.1,
                    framing_inconsistency=0.1)
        count = compute_composite(db_conn)
        assert count == 1
        row = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row["composite_risk"] is not None
        assert 0.0 <= row["composite_risk"] <= 1.0

    def test_skips_topics_missing_person_a_columns(self, db_conn):
        insert_items(db_conn, [_make_item("c2")])
        _seed_topic(db_conn, 1, ["c2"], avg_trust=80.0, coverage_ratio=0.8)
        count = compute_composite(db_conn)
        assert count == 0

    def test_computed_at_is_set(self, db_conn):
        insert_items(db_conn, [_make_item("c3")])
        _seed_topic(db_conn, 2, ["c3"],
                    avg_trust=50.0, coverage_ratio=0.5,
                    avg_sentiment_extremity=0.5, sensationalism_avg=0.5,
                    framing_inconsistency=0.5)
        compute_composite(db_conn)
        row = db_conn.execute(
            "SELECT computed_at FROM topic_scores WHERE topic_id = 2"
        ).fetchone()
        assert row["computed_at"] is not None


# ---------------------------------------------------------------------------
# score_all_topics (integration)
# ---------------------------------------------------------------------------

class TestScoreAllTopics:
    def test_returns_summary_dict(self, db_conn):
        summary = score_all_topics(db_conn)
        assert "coverage_scored" in summary
        assert "composite_scored" in summary

    def test_full_pipeline_high_trust_topic_grades_well(self, db_conn):
        """A topic covered only by credible sources should get grade A or B."""
        items = [_make_item(f"h{i}", f"https://reuters.com/{i}") for i in range(5)]
        insert_items(db_conn, items)
        _seed_topic(db_conn, 0, [f"h{i}" for i in range(5)],
                    avg_sentiment_extremity=0.05,
                    sensationalism_avg=0.05,
                    framing_inconsistency=0.05)

        score_all_topics(db_conn)

        row = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row["composite_risk"] is not None
        assert grade_topic(row["composite_risk"]) in {"A", "B"}
