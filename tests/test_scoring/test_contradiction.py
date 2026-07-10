"""Tests for src/scoring/contradiction.py.

Both the sentence-embedding model and the NLI pipeline are always mocked —
this module never hits a real model or the network in tests (see
.claude/rules/testing.md).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.scoring.contradiction import (
    _MAX_PAIRS,
    _is_contradiction,
    _select_candidate_pairs,
    _split_sentences,
    compute_fact_inconsistency,
)


def _make_scored(cleaned_text: str, source: str = "example.com", item_id: str = "a1") -> dict:
    return {
        "id": item_id,
        "title": "Article title",
        "description": None,
        "source": source,
        "url": f"https://{source}/{item_id}",
        "platform": "newsapi",
        "timestamp": "2026-05-11T10:00:00Z",
        "engagement": {"score": 0, "comments": 0},
        "cleaned_text": cleaned_text,
        "sentiment_label": "neutral",
        "sentiment_score": 0.5,
        "sentiment_extremity": 0.1,
        "sensationalism_score": 0.0,
    }


class _MockEncoder:
    """Returns a fixed vector per input text (looked up by exact text)."""

    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        return np.array([self._vectors[t] for t in texts])


class _MockPipeline:
    """Returns a fixed raw NLI result per (text, text_pair), in call order otherwise."""

    def __init__(self, responses: dict[tuple[str, str], list[dict]]):
        self._responses = responses

    def __call__(self, inputs: list[dict]) -> list[list[dict]]:
        return [self._responses[(i["text"], i["text_pair"])] for i in inputs]


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

# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        text = "Der Kanzler sprach heute Morgen. Die Minister waren anwesend."
        result = _split_sentences(text)
        assert len(result) == 2

    def test_drops_short_fragments(self):
        text = "Ja. Der Kanzler sprach heute Morgen zu allen Anwesenden."
        result = _split_sentences(text)
        assert result == ["Der Kanzler sprach heute Morgen zu allen Anwesenden."]

    def test_empty_text_returns_empty(self):
        assert _split_sentences("") == []

    def test_none_like_falsy_returns_empty(self):
        assert _split_sentences(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _is_contradiction
# ---------------------------------------------------------------------------

class TestIsContradiction:
    def test_high_contradiction_score_returns_true(self):
        assert _is_contradiction(_CONTRADICTION) is True

    def test_entailment_returns_false(self):
        assert _is_contradiction(_ENTAILMENT) is False

    def test_neutral_returns_false(self):
        assert _is_contradiction(_NEUTRAL) is False

    def test_below_threshold_contradiction_returns_false(self):
        raw = [
            {"label": "entailment", "score": 0.35},
            {"label": "neutral", "score": 0.35},
            {"label": "contradiction", "score": 0.30},
        ]
        assert _is_contradiction(raw) is False

    def test_missing_contradiction_label_returns_false(self):
        raw = [{"label": "entailment", "score": 1.0}]
        assert _is_contradiction(raw) is False

    def test_label_case_insensitive(self):
        raw = [{"label": "CONTRADICTION", "score": 0.9}]
        assert _is_contradiction(raw) is True


# ---------------------------------------------------------------------------
# _select_candidate_pairs
# ---------------------------------------------------------------------------

class TestSelectCandidatePairs:
    def test_empty_tier_returns_no_pairs(self, monkeypatch):
        model = _MockEncoder({})
        assert _select_candidate_pairs([], ["low sentence here"], model) == []
        assert _select_candidate_pairs(["high sentence here"], [], model) == []

    def test_below_similarity_floor_excluded(self, monkeypatch):
        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", lambda _t: set())
        model = _MockEncoder({
            "high one": [1.0, 0.0, 0.0],
            "low one": [0.0, 1.0, 0.0],  # orthogonal → similarity 0.0 < floor
        })
        pairs = _select_candidate_pairs(["high one"], ["low one"], model)
        assert pairs == []

    def test_above_similarity_floor_included(self, monkeypatch):
        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", lambda _t: set())
        model = _MockEncoder({
            "high one": [1.0, 0.0, 0.0],
            "low one": [0.9, 0.1, 0.0],  # near-identical → high similarity
        })
        pairs = _select_candidate_pairs(["high one"], ["low one"], model)
        assert pairs == [("high one", "low one")]

    def test_shared_entity_pairs_ranked_first(self, monkeypatch):
        # Two low sentences both clear the similarity floor against the same
        # high sentence, but only one shares a named entity — it must be
        # the one kept when max_pairs caps the result at 1.
        def _fake_entities(text: str) -> set:
            return {"alice"} if text in ("high one", "low with alice") else set()

        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", _fake_entities)
        monkeypatch.setattr("src.scoring.contradiction._MAX_PAIRS", 1)
        model = _MockEncoder({
            "high one": [1.0, 0.0, 0.0],
            "low with alice": [0.9, 0.1, 0.0],
            "low no entity": [0.95, 0.05, 0.0],  # slightly higher raw similarity
        })
        pairs = _select_candidate_pairs(
            ["high one"], ["low with alice", "low no entity"], model
        )
        assert pairs == [("high one", "low with alice")]

    def test_caps_at_max_pairs(self, monkeypatch):
        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", lambda _t: set())
        vectors = {"high one": [1.0, 0.0, 0.0]}
        low_sents = [f"low {i}" for i in range(_MAX_PAIRS + 5)]
        for s in low_sents:
            vectors[s] = [0.9, 0.1, 0.0]
        model = _MockEncoder(vectors)
        pairs = _select_candidate_pairs(["high one"], low_sents, model)
        assert len(pairs) == _MAX_PAIRS


# ---------------------------------------------------------------------------
# compute_fact_inconsistency — hand-built sentence-pair scenarios
# ---------------------------------------------------------------------------

class TestComputeFactInconsistency:
    def test_no_sentences_in_either_tier_returns_zero(self):
        assert compute_fact_inconsistency([], []) == 0.0
        assert compute_fact_inconsistency(
            [_make_scored("Ein ausreichend langer Satz mit Inhalt hier.")], []
        ) == 0.0

    def test_a_same_information_different_wording_not_flagged(self, monkeypatch):
        """(a) Same info, different wording → must NOT count as inconsistency."""
        high_text = "Der Kanzler bestätigte am Montag den Rücktritt des Ministers offiziell."
        low_text = "Am Montag räumte der Minister seinen Posten, wie der Kanzler bestätigte."

        monkeypatch.setattr("src.scoring.contradiction.get_model", lambda: _MockEncoder({
            high_text: [1.0, 0.0, 0.0],
            low_text: [0.9, 0.1, 0.0],
        }))
        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", lambda _t: set())
        monkeypatch.setattr(
            "src.scoring.contradiction._get_pipeline",
            lambda: _MockPipeline({(high_text, low_text): _ENTAILMENT}),
        )

        result = compute_fact_inconsistency(
            [_make_scored(high_text, "trusted.example")],
            [_make_scored(low_text, "tabloid.example")],
        )
        assert result == 0.0

    def test_b_genuinely_contradictory_numbers_flagged(self, monkeypatch):
        """(b) Genuinely contradictory info (different numbers for the same
        statistic) → must count as inconsistency."""
        high_text = "Das Ministerium meldete zwölf Verletzte bei dem Vorfall am Bahnhof."
        low_text = "Augenzeugen zufolge gab es beim Vorfall am Bahnhof mindestens fünfzig Verletzte."

        monkeypatch.setattr("src.scoring.contradiction.get_model", lambda: _MockEncoder({
            high_text: [1.0, 0.0, 0.0],
            low_text: [0.9, 0.1, 0.0],
        }))
        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", lambda _t: set())
        monkeypatch.setattr(
            "src.scoring.contradiction._get_pipeline",
            lambda: _MockPipeline({(high_text, low_text): _CONTRADICTION}),
        )

        result = compute_fact_inconsistency(
            [_make_scored(high_text, "trusted.example")],
            [_make_scored(low_text, "tabloid.example")],
        )
        assert result == 1.0

    def test_c_different_people_cited_no_contradiction_not_flagged(self, monkeypatch):
        """(c) Articles cite different people but do not contradict each
        other → must NOT count. This is exactly the false positive the old
        Jaccard entity-overlap metric produced (disjoint entity sets scored
        as fully inconsistent even though nothing was factually disputed)."""
        high_text = "Bundeskanzler Michael Fischer eroeffnete die Konferenz zum Klimaschutz."
        low_text = "Umweltministerin Anna Berger nahm ebenfalls an der Konferenz zum Klimaschutz teil."

        monkeypatch.setattr("src.scoring.contradiction.get_model", lambda: _MockEncoder({
            high_text: [1.0, 0.0, 0.0],
            low_text: [0.9, 0.1, 0.0],
        }))

        def _fake_entities(text: str) -> set:
            if text == high_text:
                return {"michael fischer"}
            if text == low_text:
                return {"anna berger"}
            return set()

        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", _fake_entities)
        monkeypatch.setattr(
            "src.scoring.contradiction._get_pipeline",
            lambda: _MockPipeline({(high_text, low_text): _NEUTRAL}),
        )

        result = compute_fact_inconsistency(
            [_make_scored(high_text, "trusted.example")],
            [_make_scored(low_text, "tabloid.example")],
        )
        assert result == 0.0

    def test_proportion_with_mixed_pairs(self, monkeypatch):
        """2 of 4 evaluated pairs are contradictions → 0.5."""
        high_texts = [
            "Erster Satz der hohen Vertrauensstufe mit Zahlenangabe zwoelf.",
            "Zweiter Satz der hohen Vertrauensstufe mit Zahlenangabe zwoelf.",
        ]
        low_texts = [
            "Erster Satz der niedrigen Vertrauensstufe mit Zahlenangabe fuenfzig.",
            "Zweiter Satz der niedrigen Vertrauensstufe mit Zahlenangabe fuenfzig.",
        ]
        vectors = {t: [1.0, 0.0, 0.0] for t in high_texts}
        vectors.update({t: [0.9, 0.1, 0.0] for t in low_texts})

        monkeypatch.setattr("src.scoring.contradiction.get_model", lambda: _MockEncoder(vectors))
        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", lambda _t: set())

        responses = {
            (high_texts[0], low_texts[0]): _CONTRADICTION,
            (high_texts[0], low_texts[1]): _ENTAILMENT,
            (high_texts[1], low_texts[0]): _CONTRADICTION,
            (high_texts[1], low_texts[1]): _NEUTRAL,
        }
        monkeypatch.setattr(
            "src.scoring.contradiction._get_pipeline", lambda: _MockPipeline(responses)
        )

        result = compute_fact_inconsistency(
            [_make_scored(high_texts[0], "trusted.example", "h0"),
             _make_scored(high_texts[1], "trusted.example", "h1")],
            [_make_scored(low_texts[0], "tabloid.example", "l0"),
             _make_scored(low_texts[1], "tabloid.example", "l1")],
        )
        assert result == 0.5

    def test_result_bounded_zero_to_one(self, monkeypatch):
        high_text = "Ein ausreichend langer erster Satz fuer den Test hier."
        low_text = "Ein ausreichend langer zweiter Satz fuer den Test hier."
        monkeypatch.setattr("src.scoring.contradiction.get_model", lambda: _MockEncoder({
            high_text: [1.0, 0.0, 0.0],
            low_text: [0.9, 0.1, 0.0],
        }))
        monkeypatch.setattr("src.scoring.contradiction._sentence_entities", lambda _t: set())
        monkeypatch.setattr(
            "src.scoring.contradiction._get_pipeline",
            lambda: _MockPipeline({(high_text, low_text): _CONTRADICTION}),
        )
        result = compute_fact_inconsistency(
            [_make_scored(high_text)], [_make_scored(low_text)]
        )
        assert 0.0 <= result <= 1.0
