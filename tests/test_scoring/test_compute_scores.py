"""Tests for src/scoring/compute_scores.py.

All DB interactions use a file-backed SQLite DB under tmp_path.
avg_article_risk and framing_inconsistency are inserted directly into
topic_scores to simulate run_nlp having already run.
"""

from __future__ import annotations

import pytest

from src.scoring.article_scorer import score_article
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
    def test_all_safe_signals_is_zero_risk(self):
        risk = compute_risk(
            avg_article_risk=0.0,
            framing_inconsistency=0.0,
            fact_inconsistency=0.0,
        )
        assert risk == pytest.approx(0.0)

    def test_all_max_signals_is_one_risk(self):
        risk = compute_risk(
            avg_article_risk=1.0,
            framing_inconsistency=1.0,
            fact_inconsistency=1.0,
        )
        assert risk == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        assert sum(_WEIGHTS.values()) == pytest.approx(1.0)

    def test_formula_matches_manual_calculation(self):
        risk = compute_risk(
            avg_article_risk=0.6,
            framing_inconsistency=0.4,
            fact_inconsistency=0.2,
        )
        expected = (
            0.55 * 0.6
            + 0.10 * 0.4
            + 0.35 * 0.2
        )
        assert risk == pytest.approx(expected, abs=1e-6)

    def test_higher_article_risk_raises_composite(self):
        low = compute_risk(avg_article_risk=0.1, framing_inconsistency=0.3,
                           fact_inconsistency=0.1)
        high = compute_risk(avg_article_risk=0.9, framing_inconsistency=0.3,
                            fact_inconsistency=0.1)
        assert high > low


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

    def test_misinfo_threshold_maps_to_c_d_or_f(self):
        grade = grade_topic(_MISINFO_THRESHOLD)
        assert grade in {"C", "D", "F"}


# ---------------------------------------------------------------------------
# explain_score
# ---------------------------------------------------------------------------

class TestExplainScore:
    def test_returns_all_expected_keys(self, db_conn):
        insert_items(db_conn, [_make_item("e1")])
        _seed_topic(db_conn, 0, ["e1"],
                    avg_article_risk=0.3, framing_inconsistency=0.2,
                    coverage_ratio=0.7, fact_inconsistency=0.1,
                    composite_risk=0.22)
        result = explain_score(0, db_conn)
        assert {"topic_id", "composite_risk", "grade", "contributions"} <= result.keys()

    def test_contributions_keys_present(self, db_conn):
        insert_items(db_conn, [_make_item("e2")])
        _seed_topic(db_conn, 1, ["e2"],
                    avg_article_risk=0.5, framing_inconsistency=0.5,
                    fact_inconsistency=0.5,
                    composite_risk=0.5)
        result = explain_score(1, db_conn)
        expected_keys = {
            "article_risk", "framing_inconsistency", "fact_inconsistency",
        }
        assert expected_keys == result["contributions"].keys()

    def test_contributions_sum_to_composite_risk(self, db_conn):
        insert_items(db_conn, [_make_item("e3")])
        expected_risk = compute_risk(
            avg_article_risk=0.55,
            framing_inconsistency=0.4,
            fact_inconsistency=0.2,
        )
        _seed_topic(db_conn, 2, ["e3"],
                    avg_article_risk=0.55, framing_inconsistency=0.4,
                    fact_inconsistency=0.2,
                    composite_risk=expected_risk)
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
                    avg_article_risk=0.2, framing_inconsistency=0.1,
                    coverage_ratio=0.8)
        count = compute_composite(db_conn)
        assert count == 1
        row = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row["composite_risk"] is not None
        assert 0.0 <= row["composite_risk"] <= 1.0

    def test_skips_topics_missing_avg_article_risk(self, db_conn):
        insert_items(db_conn, [_make_item("c2")])
        _seed_topic(db_conn, 1, ["c2"], framing_inconsistency=0.3, coverage_ratio=0.8)
        count = compute_composite(db_conn)
        assert count == 0

    def test_skips_topics_missing_framing_inconsistency(self, db_conn):
        insert_items(db_conn, [_make_item("c3")])
        _seed_topic(db_conn, 2, ["c3"], avg_article_risk=0.4, coverage_ratio=0.8)
        count = compute_composite(db_conn)
        assert count == 0

    def test_computed_at_is_set(self, db_conn):
        insert_items(db_conn, [_make_item("c4")])
        _seed_topic(db_conn, 3, ["c4"],
                    avg_article_risk=0.5, framing_inconsistency=0.5,
                    coverage_ratio=0.5)
        compute_composite(db_conn)
        row = db_conn.execute(
            "SELECT computed_at FROM topic_scores WHERE topic_id = 3"
        ).fetchone()
        assert row["computed_at"] is not None


# ---------------------------------------------------------------------------
# compute_composite — social track and divergence
# ---------------------------------------------------------------------------

class TestComputeCompositeSocialTrack:
    def test_social_risk_computed_when_social_signals_present(self, db_conn):
        insert_items(db_conn, [_make_item("s1")])
        _seed_topic(db_conn, 10, ["s1"],
                    avg_article_risk=0.2, framing_inconsistency=0.1,
                    coverage_ratio=0.8,
                    social_avg_trust=45.0, social_coverage_ratio=0.0,
                    social_avg_sentiment_extremity=0.7,
                    social_sensationalism_avg=0.6,
                    social_framing_inconsistency=0.5)
        compute_composite(db_conn)
        row = db_conn.execute(
            "SELECT composite_risk, social_risk, narrative_divergence FROM topic_scores WHERE topic_id = 10"
        ).fetchone()
        assert row["composite_risk"] is not None
        assert row["social_risk"] is not None
        assert 0.0 <= row["social_risk"] <= 1.0
        assert row["narrative_divergence"] is not None

    def test_social_risk_null_when_social_signals_missing(self, db_conn):
        insert_items(db_conn, [_make_item("s2")])
        _seed_topic(db_conn, 11, ["s2"],
                    avg_article_risk=0.2, framing_inconsistency=0.1,
                    coverage_ratio=0.8)
        compute_composite(db_conn)
        row = db_conn.execute(
            "SELECT social_risk, narrative_divergence FROM topic_scores WHERE topic_id = 11"
        ).fetchone()
        assert row["social_risk"] is None
        assert row["narrative_divergence"] is None

    def test_narrative_divergence_is_abs_difference(self, db_conn):
        insert_items(db_conn, [_make_item("s3")])
        _seed_topic(db_conn, 12, ["s3"],
                    avg_article_risk=0.2, framing_inconsistency=0.1,
                    coverage_ratio=0.8,
                    social_avg_trust=45.0, social_coverage_ratio=0.0,
                    social_avg_sentiment_extremity=0.7,
                    social_sensationalism_avg=0.6,
                    social_framing_inconsistency=0.5)
        compute_composite(db_conn)
        row = db_conn.execute(
            "SELECT composite_risk, social_risk, narrative_divergence FROM topic_scores WHERE topic_id = 12"
        ).fetchone()
        expected_div = abs(row["composite_risk"] - row["social_risk"])
        assert row["narrative_divergence"] == pytest.approx(expected_div, abs=1e-5)

    def test_higher_social_sensationalism_raises_social_risk(self, db_conn):
        insert_items(db_conn, [_make_item("s4"), _make_item("s5")])
        base = dict(
            avg_article_risk=0.3, framing_inconsistency=0.2, coverage_ratio=0.7,
            social_avg_trust=50.0, social_coverage_ratio=0.1,
            social_framing_inconsistency=0.3,
        )
        _seed_topic(db_conn, 20, ["s4"], **base,
                    social_avg_sentiment_extremity=0.2, social_sensationalism_avg=0.2)
        _seed_topic(db_conn, 21, ["s5"], **base,
                    social_avg_sentiment_extremity=0.8, social_sensationalism_avg=0.9)
        compute_composite(db_conn)
        row_low = db_conn.execute(
            "SELECT social_risk FROM topic_scores WHERE topic_id = 20"
        ).fetchone()
        row_high = db_conn.execute(
            "SELECT social_risk FROM topic_scores WHERE topic_id = 21"
        ).fetchone()
        assert row_high["social_risk"] > row_low["social_risk"]


# ---------------------------------------------------------------------------
# score_all_topics (integration)
# ---------------------------------------------------------------------------

class TestScoreAllTopics:
    def test_returns_summary_dict(self, db_conn):
        summary = score_all_topics(db_conn)
        assert "coverage_scored" in summary
        assert "composite_scored" in summary

    def test_full_pipeline_low_risk_topic_grades_well(self, db_conn):
        """A topic with low article risk and good framing should grade A or B."""
        items = [_make_item(f"h{i}", f"https://reuters.com/{i}") for i in range(5)]
        insert_items(db_conn, items)
        _seed_topic(db_conn, 0, [f"h{i}" for i in range(5)],
                    avg_article_risk=0.05,
                    framing_inconsistency=0.05,
                    coverage_ratio=0.9)

        score_all_topics(db_conn)

        row = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row["composite_risk"] is not None
        assert grade_topic(row["composite_risk"]) in {"A", "B"}
