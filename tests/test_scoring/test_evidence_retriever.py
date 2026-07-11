"""Tests for src/scoring/evidence_retriever.py.

All network calls (Google Fact Check API, Wikidata SPARQL) and the
sentence-embedding model are mocked — this module never hits the network or
loads a real model in tests (see .claude/rules/testing.md).
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.scoring.evidence_retriever import (
    Evidence,
    _query_corpus,
    _query_google_fact_check,
    _query_wikidata,
    _sparql_escape,
    retrieve_evidence,
)


# ---------------------------------------------------------------------------
# _sparql_escape
# ---------------------------------------------------------------------------

class TestSparqlEscape:
    def test_escapes_double_quotes(self):
        assert _sparql_escape('Say "hello"') == 'Say \\"hello\\"'

    def test_escapes_backslashes(self):
        assert _sparql_escape("a\\b") == "a\\\\b"

    def test_strips_newlines(self):
        assert "\n" not in _sparql_escape("line one\nline two")

    def test_injection_attempt_cannot_break_out_of_literal(self):
        malicious = '" }} DROP EVERYTHING SELECT ?x WHERE {{ ?x rdfs:label "'
        escaped = _sparql_escape(malicious)
        # Every double quote must be escaped — none survive unescaped.
        unescaped_quotes = escaped.replace('\\"', "")
        assert '"' not in unescaped_quotes


# ---------------------------------------------------------------------------
# _query_google_fact_check
# ---------------------------------------------------------------------------

class TestQueryGoogleFactCheck:
    def test_no_api_key_returns_none(self):
        with patch("src.utils.config.GOOGLE_FACT_CHECK_API_KEY", None):
            assert _query_google_fact_check("Irgendeine Behauptung.") is None

    def test_no_claims_in_response_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        with (
            patch("src.utils.config.GOOGLE_FACT_CHECK_API_KEY", "fake-key"),
            patch("requests.get", return_value=mock_resp),
        ):
            assert _query_google_fact_check("Irgendeine Behauptung.") is None

    def test_parses_first_claim_review(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "claims": [
                {
                    "text": "12 Verletzte bei Vorfall",
                    "claimReview": [
                        {
                            "title": "Fact check: 12 Verletzte bestätigt",
                            "url": "https://factcheck.example/12-verletzte",
                            "textualRating": "Richtig",
                        }
                    ],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        with (
            patch("src.utils.config.GOOGLE_FACT_CHECK_API_KEY", "fake-key"),
            patch("requests.get", return_value=mock_resp),
        ):
            evidence = _query_google_fact_check("12 Verletzte bei Vorfall")

        assert evidence is not None
        assert evidence.source_type == "google_factcheck"
        assert evidence.url == "https://factcheck.example/12-verletzte"
        assert evidence.rating == "Richtig"

    def test_request_exception_returns_none(self):
        with (
            patch("src.utils.config.GOOGLE_FACT_CHECK_API_KEY", "fake-key"),
            patch("requests.get", side_effect=ConnectionError("boom")),
        ):
            assert _query_google_fact_check("Irgendeine Behauptung.") is None


# ---------------------------------------------------------------------------
# _query_wikidata
# ---------------------------------------------------------------------------

class TestQueryWikidata:
    def test_empty_entities_returns_none(self):
        assert _query_wikidata([]) is None

    def test_no_bindings_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": {"bindings": []}}
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            assert _query_wikidata(["Nichtvorhandene Entität"]) is None

    def test_returns_evidence_from_first_matching_entity(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "https://www.wikidata.org/entity/Q567"},
                        "itemDescription": {"value": "deutsche Politikerin"},
                    }
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            evidence = _query_wikidata(["Angela Merkel"])

        assert evidence is not None
        assert evidence.source_type == "wikidata"
        assert evidence.text == "deutsche Politikerin"
        assert evidence.url == "https://www.wikidata.org/entity/Q567"

    def test_disambiguation_page_description_skipped_for_next_entity(self):
        disambig_resp = MagicMock()
        disambig_resp.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "https://www.wikidata.org/entity/Q1"},
                        "itemDescription": {"value": "Wikimedia-Begriffsklärungsseite"},
                    }
                ]
            }
        }
        disambig_resp.raise_for_status = MagicMock()
        good_resp = MagicMock()
        good_resp.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "https://www.wikidata.org/entity/Q2"},
                        "itemDescription": {"value": "Nachrichtenagentur"},
                    }
                ]
            }
        }
        good_resp.raise_for_status = MagicMock()
        with patch("requests.get", side_effect=[disambig_resp, good_resp]):
            evidence = _query_wikidata(["dpa", "Reuters"])

        assert evidence is not None
        assert evidence.text == "Nachrichtenagentur"

    def test_request_exception_tries_next_entity(self):
        good_resp = MagicMock()
        good_resp.json.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "https://www.wikidata.org/entity/Q1"},
                        "itemDescription": {"value": "Organisation"},
                    }
                ]
            }
        }
        good_resp.raise_for_status = MagicMock()
        with patch(
            "requests.get",
            side_effect=[ConnectionError("boom"), good_resp],
        ):
            evidence = _query_wikidata(["Erste Entität", "Zweite Entität"])

        assert evidence is not None
        assert evidence.text == "Organisation"


# ---------------------------------------------------------------------------
# _query_corpus
# ---------------------------------------------------------------------------

class _MockEncoder:
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def encode(self, texts, **kwargs):
        return np.array([self._vectors[t] for t in texts])


@pytest.fixture
def corpus_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE raw_items (id TEXT PRIMARY KEY, url TEXT, cleaned_text TEXT)")
    conn.execute("CREATE TABLE topic_sources (topic_id INTEGER, item_id TEXT)")
    yield conn
    conn.close()


class TestQueryCorpus:
    def test_no_other_articles_returns_none(self, corpus_conn):
        result = _query_corpus("Eine Behauptung.", corpus_conn, topic_id=1, exclude_item_id=None)
        assert result is None

    def test_excludes_source_article(self, corpus_conn):
        corpus_conn.execute(
            "INSERT INTO raw_items VALUES ('a1', 'https://x.example/a1', 'Ein ausreichend langer Satz zum Testen hier.')"
        )
        corpus_conn.execute("INSERT INTO topic_sources VALUES (1, 'a1')")
        corpus_conn.commit()
        result = _query_corpus("Ein ausreichend langer Satz zum Testen hier.", corpus_conn, topic_id=1, exclude_item_id="a1")
        assert result is None

    def test_returns_best_matching_sentence_above_floor(self, corpus_conn):
        corpus_conn.execute(
            "INSERT INTO raw_items VALUES ('a2', 'https://x.example/a2', "
            "'Das Ministerium bestätigte 12 Verletzte beim Vorfall am Bahnhof heute.')"
        )
        corpus_conn.execute("INSERT INTO topic_sources VALUES (1, 'a2')")
        corpus_conn.commit()

        claim_text = "12 Verletzte beim Vorfall am Bahnhof"
        matching_sentence = "Das Ministerium bestätigte 12 Verletzte beim Vorfall am Bahnhof heute."
        mock_model = _MockEncoder({
            claim_text: [1.0, 0.0, 0.0],
            matching_sentence: [0.95, 0.05, 0.0],
        })
        with patch("src.nlp.embeddings.get_model", return_value=mock_model):
            result = _query_corpus(claim_text, corpus_conn, topic_id=1, exclude_item_id=None)

        assert result is not None
        assert result.source_type == "corpus"
        assert result.url == "https://x.example/a2"

    def test_below_similarity_floor_returns_none(self, corpus_conn):
        corpus_conn.execute(
            "INSERT INTO raw_items VALUES ('a3', 'https://x.example/a3', "
            "'Ein völlig unrelated Satz über Wetter und Fußball heute Abend.')"
        )
        corpus_conn.execute("INSERT INTO topic_sources VALUES (1, 'a3')")
        corpus_conn.commit()

        claim_text = "12 Verletzte beim Vorfall am Bahnhof"
        unrelated_sentence = "Ein völlig unrelated Satz über Wetter und Fußball heute Abend."
        mock_model = _MockEncoder({
            claim_text: [1.0, 0.0, 0.0],
            unrelated_sentence: [0.0, 1.0, 0.0],
        })
        with patch("src.nlp.embeddings.get_model", return_value=mock_model):
            result = _query_corpus(claim_text, corpus_conn, topic_id=1, exclude_item_id=None)

        assert result is None


# ---------------------------------------------------------------------------
# retrieve_evidence — tier ordering
# ---------------------------------------------------------------------------

class TestRetrieveEvidence:
    def test_factcheck_hit_short_circuits_other_tiers(self):
        fc_evidence = Evidence(text="x", url="https://fc.example", source_type="google_factcheck")
        with (
            patch("src.scoring.evidence_retriever._query_google_fact_check", return_value=fc_evidence),
            patch("src.scoring.evidence_retriever._query_wikidata") as mock_wiki,
            patch("src.scoring.evidence_retriever._query_corpus") as mock_corpus,
        ):
            result = retrieve_evidence("Eine Behauptung.")

        assert result is fc_evidence
        mock_wiki.assert_not_called()
        mock_corpus.assert_not_called()

    def test_falls_through_to_wikidata_when_factcheck_empty(self):
        wiki_evidence = Evidence(text="y", url="https://wikidata.example", source_type="wikidata")
        with (
            patch("src.scoring.evidence_retriever._query_google_fact_check", return_value=None),
            patch("src.scoring.evidence_retriever._query_wikidata", return_value=wiki_evidence),
            patch("src.scoring.evidence_retriever._query_corpus") as mock_corpus,
        ):
            result = retrieve_evidence("Eine Behauptung.", entities=["Berlin"])

        assert result is wiki_evidence
        mock_corpus.assert_not_called()

    def test_falls_through_to_corpus_when_others_empty(self):
        conn = sqlite3.connect(":memory:")
        corpus_evidence = Evidence(text="z", url="https://corpus.example", source_type="corpus")
        with (
            patch("src.scoring.evidence_retriever._query_google_fact_check", return_value=None),
            patch("src.scoring.evidence_retriever._query_wikidata", return_value=None),
            patch("src.scoring.evidence_retriever._query_corpus", return_value=corpus_evidence),
        ):
            result = retrieve_evidence("Eine Behauptung.", conn=conn, topic_id=1)
        conn.close()

        assert result is corpus_evidence

    def test_corpus_tier_skipped_without_conn_or_topic_id(self):
        with (
            patch("src.scoring.evidence_retriever._query_google_fact_check", return_value=None),
            patch("src.scoring.evidence_retriever._query_wikidata", return_value=None),
            patch("src.scoring.evidence_retriever._query_corpus") as mock_corpus,
        ):
            result = retrieve_evidence("Eine Behauptung.")

        assert result is None
        mock_corpus.assert_not_called()

    def test_no_tier_finds_anything_returns_none(self):
        with (
            patch("src.scoring.evidence_retriever._query_google_fact_check", return_value=None),
            patch("src.scoring.evidence_retriever._query_wikidata", return_value=None),
        ):
            result = retrieve_evidence("Eine Behauptung.")
        assert result is None
