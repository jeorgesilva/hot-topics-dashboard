"""Full pipeline integration test.

Verifies that the end-to-end scoring pipeline — scraper output →
topic_scores table — works correctly with all fields present,
including the new attribution_vagueness and fact_inconsistency columns.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest

from src.scoring.compute_scores import _MISINFO_THRESHOLD, grade_topic, score_all_topics
from src.utils.db import init_db, insert_items
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_item(item_id: str, url: str) -> RawItem:
    return {
        "id": item_id,
        "title": f"Article {item_id}",
        "description": None,
        "source": url.split("/")[2],
        "url": url,
        "platform": "newsapi",
        "timestamp": "2026-05-19T10:00:00Z",
        "engagement": {"score": 0, "comments": 0},
    }


@pytest.fixture
def db_conn(tmp_path):
    conn = init_db(tmp_path / "integration.db")
    yield conn
    conn.close()


def _seed_topic(
    conn,
    topic_id: int,
    label: str,
    articles: list[tuple[str, str]],
    person_a: dict,
) -> None:
    items = [_make_item(iid, url) for iid, url in articles]
    insert_items(conn, items)
    with conn:
        conn.execute(
            "INSERT INTO topics (id, label, created_at, item_count) VALUES (?, ?, '2026-05-19T00:00:00Z', ?)",
            (topic_id, label, len(articles)),
        )
        conn.executemany(
            "INSERT INTO topic_sources (topic_id, item_id) VALUES (?, ?)",
            [(topic_id, iid) for iid, _ in articles],
        )
        if person_a:
            cols = ", ".join(person_a.keys())
            placeholders = ", ".join("?" * len(person_a))
            conn.execute(
                f"INSERT INTO topic_scores (topic_id, {cols}) VALUES (?, {placeholders})",
                (topic_id, *person_a.values()),
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_all_topics_receive_composite_risk(self, db_conn):
        _seed_topic(db_conn, 0, "High trust topic", [
            ("h1", "https://reuters.com/a"),
            ("h2", "https://bbc.com/b"),
            ("h3", "https://apnews.com/c"),
        ], {"avg_sentiment_extremity": 0.05, "sensationalism_avg": 0.05, "framing_inconsistency": 0.05, "avg_article_risk": 0.10})

        _seed_topic(db_conn, 1, "Low trust topic", [
            ("l1", "https://infowars.com/a"),
            ("l2", "https://naturalnews.com/b"),
        ], {"avg_sentiment_extremity": 0.90, "sensationalism_avg": 0.88, "framing_inconsistency": 0.85, "avg_article_risk": 0.88})

        summary = score_all_topics(db_conn)

        assert summary["coverage_scored"] == 2
        assert summary["composite_scored"] == 2

        rows = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE composite_risk IS NOT NULL"
        ).fetchall()
        assert len(rows) == 2
        assert all(0.0 <= r["composite_risk"] <= 1.0 for r in rows)

    def test_high_trust_topic_grades_better_than_low_trust(self, db_conn):
        _seed_topic(db_conn, 0, "Credible topic", [
            ("c1", "https://reuters.com/a"),
            ("c2", "https://bbc.com/b"),
            ("c3", "https://apnews.com/c"),
            ("c4", "https://npr.org/d"),
        ], {"avg_sentiment_extremity": 0.05, "sensationalism_avg": 0.05, "framing_inconsistency": 0.05, "avg_article_risk": 0.10})

        _seed_topic(db_conn, 1, "Conspiracy topic", [
            ("x1", "https://infowars.com/a"),
            ("x2", "https://naturalnews.com/b"),
            ("x3", "https://beforeitsnews.com/c"),
        ], {"avg_sentiment_extremity": 0.92, "sensationalism_avg": 0.90, "framing_inconsistency": 0.88, "avg_article_risk": 0.90})

        score_all_topics(db_conn)

        rows = {
            r["topic_id"]: r["composite_risk"]
            for r in db_conn.execute(
                "SELECT topic_id, composite_risk FROM topic_scores"
            ).fetchall()
        }
        assert rows[0] < rows[1], "Credible topic should have lower risk than conspiracy topic"
        assert grade_topic(rows[0]) in {"A", "B"}
        assert grade_topic(rows[1]) in {"D", "F"}

    def test_conspiracy_topic_exceeds_misinfo_threshold(self, db_conn):
        _seed_topic(db_conn, 0, "Conspiracy", [
            ("x1", "https://infowars.com/a"),
            ("x2", "https://naturalnews.com/b"),
        ], {"avg_sentiment_extremity": 0.95, "sensationalism_avg": 0.92, "framing_inconsistency": 0.91, "avg_article_risk": 0.92})

        score_all_topics(db_conn)

        row = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row["composite_risk"] >= _MISINFO_THRESHOLD

    def test_attribution_and_fact_columns_default_to_zero(self, db_conn):
        _seed_topic(db_conn, 0, "No person-a extras", [
            ("a1", "https://reuters.com/a"),
        ], {"avg_sentiment_extremity": 0.1, "sensationalism_avg": 0.1, "framing_inconsistency": 0.1, "avg_article_risk": 0.15})

        score_all_topics(db_conn)

        row = db_conn.execute(
            "SELECT composite_risk, attribution_vagueness, fact_inconsistency "
            "FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row["composite_risk"] is not None
        # columns are NULL (not yet populated by Person A) but scoring treats them as 0.0
        assert row["attribution_vagueness"] is None
        assert row["fact_inconsistency"] is None

    def test_attribution_and_fact_signals_increase_risk(self, db_conn):
        """Adding attribution_vagueness / fact_inconsistency must raise composite_risk."""
        _seed_topic(db_conn, 0, "Without extra signals", [
            ("b1", "https://reuters.com/a"),
        ], {"avg_sentiment_extremity": 0.5, "sensationalism_avg": 0.5, "framing_inconsistency": 0.5, "avg_article_risk": 0.5})

        _seed_topic(db_conn, 1, "With extra signals", [
            ("b2", "https://reuters.com/b"),
        ], {
            "avg_sentiment_extremity": 0.5,
            "sensationalism_avg": 0.5,
            "framing_inconsistency": 0.5,
            "avg_article_risk": 0.5,
            "attribution_vagueness": 1.0,
            "fact_inconsistency": 1.0,
        })

        score_all_topics(db_conn)

        risks = {
            r["topic_id"]: r["composite_risk"]
            for r in db_conn.execute(
                "SELECT topic_id, composite_risk FROM topic_scores"
            ).fetchall()
        }
        assert risks[1] > risks[0]

    def test_topic_without_person_a_signals_is_skipped(self, db_conn):
        _seed_topic(db_conn, 0, "Incomplete", [
            ("s1", "https://reuters.com/a"),
        ], {})  # no Person A signals

        summary = score_all_topics(db_conn)
        assert summary["composite_scored"] == 0

    def test_computed_at_is_populated(self, db_conn):
        _seed_topic(db_conn, 0, "Timed topic", [
            ("t1", "https://reuters.com/a"),
        ], {"avg_sentiment_extremity": 0.2, "sensationalism_avg": 0.2, "framing_inconsistency": 0.2, "avg_article_risk": 0.25})

        score_all_topics(db_conn)

        row = db_conn.execute(
            "SELECT computed_at FROM topic_scores WHERE topic_id = 0"
        ).fetchone()
        assert row["computed_at"] is not None

    def test_pipeline_is_idempotent(self, db_conn):
        _seed_topic(db_conn, 0, "Idempotent topic", [
            ("i1", "https://reuters.com/a"),
            ("i2", "https://bbc.com/b"),
        ], {"avg_sentiment_extremity": 0.3, "sensationalism_avg": 0.3, "framing_inconsistency": 0.3, "avg_article_risk": 0.35})

        score_all_topics(db_conn)
        risk_first = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE topic_id = 0"
        ).fetchone()["composite_risk"]

        score_all_topics(db_conn)
        risk_second = db_conn.execute(
            "SELECT composite_risk FROM topic_scores WHERE topic_id = 0"
        ).fetchone()["composite_risk"]

        assert risk_first == pytest.approx(risk_second)
        assert db_conn.execute("SELECT COUNT(*) FROM topic_scores").fetchone()[0] == 1


class TestAppStartup:
    def test_app_module_syntax_is_valid(self):
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", "src/dashboard/app.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_app_starts_without_error(self):
        # Find a free ephemeral port so the test never conflicts with other processes.
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]

        proc = subprocess.Popen(
            [
                sys.executable, "-m", "streamlit", "run",
                "src/dashboard/app.py",
                "--server.headless", "true",
                f"--server.port={port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # Poll until the HTTP port is accepting connections (max 20 s).
            deadline = time.monotonic() + 20
            started = False
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break  # process died — assertion below will catch it
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        started = True
                        break
                except OSError:
                    time.sleep(0.3)

            assert proc.poll() is None, (
                "Streamlit process exited unexpectedly before becoming ready."
            )
            assert started, f"Streamlit did not bind to port {port} within 20 s."
        finally:
            proc.terminate()
            proc.wait(timeout=5)
