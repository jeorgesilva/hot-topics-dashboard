"""Tests for src/scoring/claim_verifier.py.

The NLI pipeline and evidence retrieval are always mocked — this module
never hits a real model or the network in tests (see .claude/rules/testing.md).
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest

from src.scoring.claim_verifier import (
    save_verification,
    verify_article_claims,
    verify_claim,
)
from src.scoring.evidence_retriever import Evidence
from src.utils.db import init_db

_ENTAILMENT = [
    {"label": "entailment", "score": 0.90},
    {"label": "neutral", "score": 0.08},
    {"label": "contradiction", "score": 0.02},
]
_NEUTRAL = [
    {"label": "entailment", "score": 0.10},
    {"label": "neutral", "score": 0.85},
    {"label": "contradiction", "score": 0.05},
]
_CONTRADICTION = [
    {"label": "entailment", "score": 0.05},
    {"label": "neutral", "score": 0.10},
    {"label": "contradiction", "score": 0.85},
]


class _MockPipeline:
    def __init__(self, response: list[dict]):
        self._response = response

    def __call__(self, inputs):
        return [self._response for _ in inputs]


# ---------------------------------------------------------------------------
# verify_claim
# ---------------------------------------------------------------------------

class TestVerifyClaim:
    def test_no_evidence_returns_not_enough_evidence(self):
        result = verify_claim("Eine Behauptung.", None)
        assert result["verdict"] == "not_enough_evidence"
        assert result["confidence"] == 0.0
        assert result["evidence_text"] is None
        assert result["evidence_url"] is None

    def test_high_entailment_returns_supported(self):
        evidence = Evidence(text="Bestätigender Text.", url="https://x.example", source_type="wikidata")
        with patch("src.scoring.claim_verifier.get_pipeline", return_value=_MockPipeline(_ENTAILMENT)):
            result = verify_claim("Eine Behauptung.", evidence)
        assert result["verdict"] == "supported"
        assert result["confidence"] == 0.90
        assert result["evidence_url"] == "https://x.example"
        assert result["evidence_source_type"] == "wikidata"

    def test_high_contradiction_returns_refuted(self):
        evidence = Evidence(text="Widersprechender Text.", url="https://y.example", source_type="corpus")
        with patch("src.scoring.claim_verifier.get_pipeline", return_value=_MockPipeline(_CONTRADICTION)):
            result = verify_claim("Eine Behauptung.", evidence)
        assert result["verdict"] == "refuted"
        assert result["confidence"] == 0.85

    def test_neutral_returns_not_enough_evidence(self):
        evidence = Evidence(text="Neutraler Text.", url="https://z.example", source_type="google_factcheck")
        with patch("src.scoring.claim_verifier.get_pipeline", return_value=_MockPipeline(_NEUTRAL)):
            result = verify_claim("Eine Behauptung.", evidence)
        assert result["verdict"] == "not_enough_evidence"
        assert result["evidence_text"] == "Neutraler Text."  # evidence still attached for display


# ---------------------------------------------------------------------------
# save_verification
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)
    yield conn
    conn.close()


class TestSaveVerification:
    def test_persists_row(self, db_conn):
        verification = {
            "claim_text": "Eine Behauptung.",
            "verdict": "supported",
            "evidence_text": "Beweistext.",
            "evidence_url": "https://x.example",
            "evidence_source_type": "wikidata",
            "confidence": 0.9,
        }
        save_verification(db_conn, topic_id=1, item_id="a1", verification=verification)

        row = db_conn.execute("SELECT * FROM claim_verifications").fetchone()
        assert row["claim_text"] == "Eine Behauptung."
        assert row["verdict"] == "supported"
        assert row["evidence_url"] == "https://x.example"
        assert row["evidence_snippet"] == "Beweistext."
        assert row["confidence"] == 0.9
        assert row["topic_id"] == 1
        assert row["item_id"] == "a1"
        assert row["checked_at"] is not None


# ---------------------------------------------------------------------------
# verify_article_claims — end-to-end with everything mocked
# ---------------------------------------------------------------------------

class TestVerifyArticleClaims:
    def test_extracts_verifies_and_persists(self, db_conn):
        text = "Das Ministerium bestätigte 12 Verletzte beim Vorfall am Bahnhof heute."
        evidence = Evidence(text="Bestätigender Text.", url="https://x.example", source_type="wikidata")

        with (
            patch("src.scoring.claim_verifier.retrieve_evidence", return_value=evidence),
            patch("src.scoring.claim_verifier.get_pipeline", return_value=_MockPipeline(_ENTAILMENT)),
        ):
            results = verify_article_claims("a1", text, db_conn, topic_id=1)

        assert len(results) == 1
        assert results[0]["verdict"] == "supported"

        row = db_conn.execute("SELECT * FROM claim_verifications").fetchone()
        assert row["item_id"] == "a1"
        assert row["topic_id"] == 1

    def test_no_claims_persists_nothing(self, db_conn):
        text = "Die Stimmung war gut. Alle waren zufrieden heute Abend."
        results = verify_article_claims("a2", text, db_conn, topic_id=1)
        assert results == []
        assert db_conn.execute("SELECT COUNT(*) FROM claim_verifications").fetchone()[0] == 0

    def test_persist_false_skips_db_write(self, db_conn):
        text = "Das Ministerium bestätigte 12 Verletzte beim Vorfall am Bahnhof heute."
        with (
            patch("src.scoring.evidence_retriever.retrieve_evidence", return_value=None),
            patch("src.scoring.claim_verifier.get_pipeline", return_value=_MockPipeline(_ENTAILMENT)),
        ):
            results = verify_article_claims("a3", text, db_conn, topic_id=1, persist=False)

        assert len(results) == 1
        assert db_conn.execute("SELECT COUNT(*) FROM claim_verifications").fetchone()[0] == 0
