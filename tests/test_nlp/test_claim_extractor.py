"""Tests for src/nlp/claim_extractor.py.

Uses the real spaCy model for entity extraction (a local dependency, not a
network call — consistent with tests/test_nlp/test_ner.py), so no mocking
of ner.extract_entities is needed here.
"""
from __future__ import annotations

from src.nlp.claim_extractor import (
    _is_check_worthy,
    _split_sentences,
    extract_claims,
)


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        text = "Der Kanzler sprach heute Morgen. Die Minister waren anwesend."
        assert len(_split_sentences(text)) == 2

    def test_drops_short_fragments(self):
        text = "Ja. Der Kanzler sprach heute Morgen zu allen Anwesenden."
        assert _split_sentences(text) == ["Der Kanzler sprach heute Morgen zu allen Anwesenden."]

    def test_empty_text_returns_empty(self):
        assert _split_sentences("") == []


class TestIsCheckWorthy:
    def test_number_makes_sentence_check_worthy(self):
        assert _is_check_worthy("Bei dem Vorfall wurden 12 Menschen verletzt.") is True

    def test_spelled_out_number_without_marker_not_check_worthy(self):
        assert _is_check_worthy("Bei dem Vorfall wurden zwölf Menschen verletzt.") is False

    def test_date_makes_sentence_check_worthy(self):
        assert _is_check_worthy("Der Vorfall ereignete sich am 12. Januar 2026 in Berlin.") is True

    def test_factual_attribution_marker_makes_sentence_check_worthy(self):
        assert _is_check_worthy("Das Ministerium bestätigte den Vorfall am Bahnhof.") is True

    def test_plain_sentence_without_number_date_or_marker_not_check_worthy(self):
        assert _is_check_worthy("Die Stimmung in der Stadt war gedrückt und traurig.") is False

    def test_opinion_marker_vetoes_even_with_number(self):
        assert _is_check_worthy("Kritiker glauben, dass 12 weitere Vorfälle folgen könnten.") is False

    def test_speculation_marker_vetoes_sentence(self):
        assert _is_check_worthy("Möglicherweise waren mehr als 50 Menschen betroffen.") is False

    def test_hedged_attribution_still_vetoed(self):
        assert _is_check_worthy("Gerüchten zufolge sollen es 100 Verletzte gewesen sein.") is False


class TestExtractClaims:
    def test_extracts_check_worthy_sentences_only(self):
        text = (
            "Das Ministerium bestätigte 12 Verletzte bei dem Vorfall am Bahnhof. "
            "Die Stimmung in der Stadt war gedrückt und traurig. "
            "Kritiker glauben, dass es noch schlimmer werden könnte."
        )
        claims = extract_claims(text)
        assert len(claims) == 1
        assert "12 Verletzte" in claims[0]["text"]

    def test_claim_has_number_and_date_flags(self):
        text = "Am 12. Januar 2026 meldete die Polizei 50 Festnahmen in der Innenstadt."
        claims = extract_claims(text)
        assert len(claims) == 1
        assert claims[0]["has_number"] is True
        assert claims[0]["has_date"] is True

    def test_claim_includes_entities(self):
        text = "Bundeskanzler Michael Fischer bestätigte 12 Verletzte beim Vorfall in Berlin."
        claims = extract_claims(text)
        assert len(claims) == 1
        assert claims[0]["entities"]  # at least one PER/ORG/LOC extracted

    def test_respects_max_claims_cap(self):
        sentence = "Das Ministerium bestätigte {} Verletzte beim Vorfall am Bahnhof heute."
        text = " ".join(sentence.format(n) for n in range(1, 10))
        claims = extract_claims(text, max_claims=3)
        assert len(claims) == 3

    def test_empty_text_returns_no_claims(self):
        assert extract_claims("") == []

    def test_no_check_worthy_sentences_returns_empty(self):
        text = "Die Stimmung war gut. Alle waren zufrieden und entspannt heute Abend."
        assert extract_claims(text) == []
