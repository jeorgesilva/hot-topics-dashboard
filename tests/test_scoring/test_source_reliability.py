"""Tests for src/scoring/source_reliability.py"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.scoring.source_reliability import (
    ReliabilityState,
    _posterior_weight,
    _prior_confidence,
    get_reliability_state,
    init_cache,
    resolve_reliability,
    sync_from_claim_verifications,
    update_reliability,
)
from src.utils.db import init_db

_NOW = datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# init_cache
# ---------------------------------------------------------------------------


class TestInitCache:
    def test_creates_tables(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "source_reliability" in tables
        assert "source_reliability_applied" in tables
        conn.close()

    def test_idempotent(self):
        conn = sqlite3.connect(":memory:")
        init_cache(conn)
        init_cache(conn)
        conn.close()


# ---------------------------------------------------------------------------
# update_reliability
# ---------------------------------------------------------------------------


class TestUpdateReliability:
    def test_first_supported_verdict_moves_score_above_neutral(self, mem_conn):
        state = update_reliability(mem_conn, "trusted.de", "supported", 0.9, at=_iso(_NOW))
        assert state["reliability_score"] > 50.0
        assert state["verdict_count"] == 1

    def test_first_refuted_verdict_moves_score_below_neutral(self, mem_conn):
        state = update_reliability(mem_conn, "sketchy.de", "refuted", 0.9, at=_iso(_NOW))
        assert state["reliability_score"] < 50.0

    def test_not_enough_evidence_does_not_move_score(self, mem_conn):
        state = update_reliability(
            mem_conn, "neutral.de", "not_enough_evidence", 0.5, at=_iso(_NOW)
        )
        assert state["reliability_score"] == 50.0
        assert state["weight_total"] == 0.0
        # verdict_count still increments — it's observed, just carries no weight.
        assert state["verdict_count"] == 1

    def test_not_enough_evidence_after_history_leaves_score_unchanged(self, mem_conn):
        update_reliability(mem_conn, "trusted.de", "supported", 0.9, at=_iso(_NOW))
        before = get_reliability_state(mem_conn, "trusted.de")
        after = update_reliability(
            mem_conn,
            "trusted.de",
            "not_enough_evidence",
            0.5,
            at=_iso(_NOW + timedelta(hours=1)),
        )
        assert after["reliability_score"] == before["reliability_score"]
        assert after["verdict_count"] == before["verdict_count"] + 1

    def test_repeated_supported_verdicts_converge_upward(self, mem_conn):
        scores = []
        for i in range(6):
            state = update_reliability(
                mem_conn, "reliable.de", "supported", 0.9, at=_iso(_NOW + timedelta(days=i))
            )
            scores.append(state["reliability_score"])
        # Monotonically non-decreasing and trending toward the ceiling.
        assert all(b >= a for a, b in zip(scores, scores[1:]))
        assert scores[-1] > 80.0

    def test_repeated_refuted_verdicts_converge_downward(self, mem_conn):
        scores = []
        for i in range(6):
            state = update_reliability(
                mem_conn, "unreliable.de", "refuted", 0.9, at=_iso(_NOW + timedelta(days=i))
            )
            scores.append(state["reliability_score"])
        assert all(b <= a for a, b in zip(scores, scores[1:]))
        assert scores[-1] < 20.0

    def test_recent_verdict_outweighs_old_opposite_verdict(self, mem_conn):
        # A strong "refuted" long ago, then a strong "supported" much more
        # recently — decay should let the recent verdict dominate.
        old = _NOW - timedelta(days=180)
        update_reliability(mem_conn, "flipped.de", "refuted", 0.95, at=_iso(old))
        state = update_reliability(mem_conn, "flipped.de", "supported", 0.95, at=_iso(_NOW))
        assert state["reliability_score"] > 50.0

    def test_mixed_signals_land_between_extremes(self, mem_conn):
        update_reliability(mem_conn, "mixed.de", "supported", 0.8, at=_iso(_NOW))
        state = update_reliability(
            mem_conn, "mixed.de", "refuted", 0.8, at=_iso(_NOW + timedelta(hours=1))
        )
        assert 20.0 < state["reliability_score"] < 80.0

    def test_posterior_weight_increases_with_verdict_count(self, mem_conn):
        weights = []
        for i in range(8):
            state = update_reliability(
                mem_conn, "growing.de", "supported", 0.9, at=_iso(_NOW + timedelta(days=i))
            )
            weights.append(_posterior_weight(state["weight_total"]))
        assert weights[-1] > weights[0]
        assert weights[-1] <= 1.0

    def test_domain_normalized_across_calls(self, mem_conn):
        update_reliability(mem_conn, "WWW.Example.DE", "supported", 0.9, at=_iso(_NOW))
        state = get_reliability_state(mem_conn, "example.de")
        assert state is not None
        assert state["domain"] == "example.de"


# ---------------------------------------------------------------------------
# get_reliability_state
# ---------------------------------------------------------------------------


class TestGetReliabilityState:
    def test_returns_none_for_unknown_domain(self, mem_conn):
        assert get_reliability_state(mem_conn, "never-seen.de") is None


# ---------------------------------------------------------------------------
# _prior_confidence
# ---------------------------------------------------------------------------


class TestPriorConfidence:
    def test_verified_csv_domain_has_high_confidence(self, mem_conn):
        # apnews.com is a source=MBFC, confidence=low entry in the real CSV;
        # patch the module-level DB so this test doesn't depend on CSV contents.
        with patch(
            "src.scoring.source_reliability._CONFIDENCE_DB", {"knownverified.de": "verified"}
        ):
            assert _prior_confidence("knownverified.de", mem_conn) == pytest.approx(0.9)

    def test_low_confidence_csv_domain(self, mem_conn):
        with patch("src.scoring.source_reliability._CONFIDENCE_DB", {"knownlow.de": "low"}):
            assert _prior_confidence("knownlow.de", mem_conn) == pytest.approx(0.6)

    def test_dynamically_resolved_domain(self, mem_conn):
        mem_conn.execute(
            "INSERT INTO domain_trust_cache (domain, trust_score, method, cached_at)"
            " VALUES ('dynamic.de', 60.0, 'live:x', ?)",
            (_iso(_NOW),),
        )
        mem_conn.commit()
        with patch("src.scoring.source_reliability._CONFIDENCE_DB", {}):
            assert _prior_confidence("dynamic.de", mem_conn) == pytest.approx(0.4)

    def test_completely_unknown_domain_has_low_confidence(self, mem_conn):
        with patch("src.scoring.source_reliability._CONFIDENCE_DB", {}):
            assert _prior_confidence("totally-unknown.de", mem_conn) == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# resolve_reliability
# ---------------------------------------------------------------------------


class TestResolveReliability:
    def test_no_history_falls_back_to_prior_score(self, mem_conn):
        with patch("src.scoring.source_trust.get_trust_score", return_value=70.0):
            estimate = resolve_reliability("no-history.de", mem_conn)
        assert estimate["trust_score"] == 70.0
        assert estimate["posterior_weight"] == 0.0
        assert estimate["verdict_count"] == 0

    def test_unknown_domain_no_history_has_low_confidence(self, mem_conn):
        with (
            patch("src.scoring.source_trust.get_trust_score", return_value=45.0),
            patch("src.scoring.source_reliability._CONFIDENCE_DB", {}),
        ):
            estimate = resolve_reliability("mystery.de", mem_conn)
        assert estimate["confidence"] < 0.3

    def test_verified_domain_no_history_has_higher_confidence(self, mem_conn):
        with (
            patch("src.scoring.source_trust.get_trust_score", return_value=90.0),
            patch(
                "src.scoring.source_reliability._CONFIDENCE_DB",
                {"verified.de": "verified"},
            ),
        ):
            estimate = resolve_reliability("verified.de", mem_conn)
        assert estimate["confidence"] == pytest.approx(0.9)

    def test_heavy_history_shifts_blended_score_toward_posterior(self, mem_conn):
        for i in range(10):
            update_reliability(
                mem_conn, "shifted.de", "refuted", 0.95, at=_iso(_NOW + timedelta(days=i))
            )
        with patch("src.scoring.source_trust.get_trust_score", return_value=90.0):
            estimate = resolve_reliability("shifted.de", mem_conn)
        # Prior said "very trustworthy" (90) but a strong refuted history
        # should pull the blended score well below that.
        assert estimate["trust_score"] < 50.0
        assert estimate["posterior_weight"] == pytest.approx(1.0)

    def test_confidence_rises_with_accumulated_verdicts(self, mem_conn):
        with (
            patch("src.scoring.source_trust.get_trust_score", return_value=50.0),
            patch("src.scoring.source_reliability._CONFIDENCE_DB", {}),
        ):
            before = resolve_reliability("accruing.de", mem_conn)["confidence"]
        for i in range(5):
            update_reliability(
                mem_conn, "accruing.de", "supported", 0.9, at=_iso(_NOW + timedelta(days=i))
            )
        with (
            patch("src.scoring.source_trust.get_trust_score", return_value=50.0),
            patch("src.scoring.source_reliability._CONFIDENCE_DB", {}),
        ):
            after = resolve_reliability("accruing.de", mem_conn)["confidence"]
        assert after > before


# ---------------------------------------------------------------------------
# sync_from_claim_verifications
# ---------------------------------------------------------------------------


def _insert_raw_item(conn: sqlite3.Connection, item_id: str, url: str) -> None:
    conn.execute(
        """
        INSERT INTO raw_items (id, title, source, url, platform, timestamp, engagement_json)
        VALUES (?, 'title', 'source', ?, 'rss', ?, '{}')
        """,
        (item_id, url, _iso(_NOW)),
    )
    conn.commit()


def _insert_verification(
    conn: sqlite3.Connection, item_id: str, verdict: str, confidence: float, checked_at: str
) -> None:
    conn.execute(
        """
        INSERT INTO claim_verifications
            (topic_id, item_id, claim_text, verdict, evidence_url, evidence_snippet, confidence, checked_at)
        VALUES (NULL, ?, 'claim', ?, NULL, NULL, ?, ?)
        """,
        (item_id, verdict, confidence, checked_at),
    )
    conn.commit()


class TestSyncFromClaimVerifications:
    def test_applies_new_verdicts_and_updates_domain(self, mem_conn):
        _insert_raw_item(mem_conn, "a1", "https://example.de/article")
        _insert_verification(mem_conn, "a1", "supported", 0.9, _iso(_NOW))

        applied = sync_from_claim_verifications(mem_conn)

        assert applied == 1
        state = get_reliability_state(mem_conn, "example.de")
        assert state is not None
        assert state["reliability_score"] > 50.0

    def test_idempotent_on_second_call(self, mem_conn):
        _insert_raw_item(mem_conn, "a1", "https://example.de/article")
        _insert_verification(mem_conn, "a1", "supported", 0.9, _iso(_NOW))

        sync_from_claim_verifications(mem_conn)
        first_state = get_reliability_state(mem_conn, "example.de")
        applied_again = sync_from_claim_verifications(mem_conn)
        second_state = get_reliability_state(mem_conn, "example.de")

        assert applied_again == 0
        assert first_state == second_state

    def test_only_processes_unapplied_rows(self, mem_conn):
        _insert_raw_item(mem_conn, "a1", "https://example.de/article-1")
        _insert_raw_item(mem_conn, "a2", "https://example.de/article-2")
        _insert_verification(mem_conn, "a1", "supported", 0.9, _iso(_NOW))

        sync_from_claim_verifications(mem_conn)
        _insert_verification(mem_conn, "a2", "supported", 0.9, _iso(_NOW + timedelta(hours=1)))
        applied = sync_from_claim_verifications(mem_conn)

        assert applied == 1
        state = get_reliability_state(mem_conn, "example.de")
        assert state["verdict_count"] == 2
