"""Tests for src/scoring/framing.py — sentence-transformers is always mocked."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.scoring.framing import (
    FramingResult,
    _cosine_similarity,
    _entity_overlap_score,
    _mean_vector,
    compute_framing,
)
from src.scoring.sentiment import ScoredArticle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_scored(
    item_id: str,
    source: str,
    cleaned_text: str = "Some article text about the topic.",
    url: str | None = None,
) -> ScoredArticle:
    return {
        "id": item_id,
        "title": "Article title",
        "description": None,
        "source": source,
        "url": url or f"https://example.com/{item_id}",
        "platform": "newsapi",
        "timestamp": "2026-05-11T10:00:00Z",
        "engagement": {"score": 0, "comments": 0},
        "cleaned_text": cleaned_text,
        "sentiment_label": "neutral",
        "sentiment_score": 0.5,
        "sentiment_extremity": 0.1,
        "sensationalism_score": 0.0,
    }


def _trust(scores: dict[str, float]):
    return scores


class _MockEncoder:
    """Returns pre-defined embeddings per call, in call order."""

    def __init__(self, *vectors: list[float]):
        self._queue: list[list[float]] = list(vectors)
        self._offset = 0

    def encode(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        batch = self._queue[self._offset: self._offset + n]
        self._offset += n
        return np.array(batch)


_EMPTY_NER = {"persons": [], "organizations": [], "locations": [], "events": []}


@pytest.fixture
def empty_ner(monkeypatch):
    """Stub NER so compute_framing tests stay fast and isolated from spaCy."""
    monkeypatch.setattr("src.scoring.framing.extract_entities", lambda _text: _EMPTY_NER)


@pytest.fixture
def identical_model(monkeypatch, empty_ner):
    """Both tiers get the same embedding → inconsistency ≈ 0."""
    vec = [1.0, 0.0, 0.0]
    encoder = _MockEncoder(*([vec] * 20))
    monkeypatch.setattr("src.scoring.framing._get_model", lambda: encoder)


@pytest.fixture
def orthogonal_model(monkeypatch, empty_ner):
    """High-trust gets [1,0,0], low-trust gets [0,1,0] → cosine distance = 1."""
    high_vecs = [[1.0, 0.0, 0.0]] * 3
    low_vecs = [[0.0, 1.0, 0.0]] * 3
    encoder = _MockEncoder(*high_vecs, *low_vecs)
    monkeypatch.setattr("src.scoring.framing._get_model", lambda: encoder)


@pytest.fixture
def mixed_model(monkeypatch, empty_ner):
    """Moderate divergence: high=[1,0,0], low=[0.5,0.5,0]."""
    high_vecs = [[1.0, 0.0, 0.0]] * 2
    low_vecs = [[0.5, 0.5, 0.0]] * 2
    encoder = _MockEncoder(*high_vecs, *low_vecs)
    monkeypatch.setattr("src.scoring.framing._get_model", lambda: encoder)


# ---------------------------------------------------------------------------
# _cosine_similarity unit tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_returns_float(self):
        assert isinstance(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), float)


# ---------------------------------------------------------------------------
# _mean_vector unit tests
# ---------------------------------------------------------------------------

class TestMeanVector:
    def test_single_vector_returned_unchanged(self):
        v = [1.0, 2.0, 3.0]
        assert _mean_vector([v]) == v

    def test_two_vectors_averaged(self):
        a = [0.0, 0.0]
        b = [2.0, 4.0]
        result = _mean_vector([a, b])
        assert result == [1.0, 2.0]

    def test_empty_returns_empty(self):
        assert _mean_vector([]) == []


# ---------------------------------------------------------------------------
# compute_framing integration tests
# ---------------------------------------------------------------------------

class TestComputeFraming:
    def test_empty_articles_returns_fallback(self):
        result = compute_framing([], {})
        assert result["framing_inconsistency"] == 0.0
        assert result["high_trust_articles"] == []
        assert result["low_trust_articles"] == []

    def test_all_high_trust_returns_fallback(self):
        articles = [_make_scored(f"a{i}", "bbc.com") for i in range(3)]
        result = compute_framing(articles, {"bbc.com": 90.0})
        assert result["framing_inconsistency"] == 0.0
        assert len(result["high_trust_articles"]) == 3
        assert result["low_trust_articles"] == []

    def test_all_low_trust_returns_fallback(self):
        articles = [_make_scored(f"a{i}", "tabloid.net") for i in range(3)]
        result = compute_framing(articles, {"tabloid.net": 20.0})
        assert result["framing_inconsistency"] == 0.0
        assert result["high_trust_articles"] == []
        assert len(result["low_trust_articles"]) == 3

    def test_identical_framing_near_zero(self, identical_model):
        articles = (
            [_make_scored(f"h{i}", "bbc.com") for i in range(2)]
            + [_make_scored(f"l{i}", "tabloid.net") for i in range(2)]
        )
        trust = {"bbc.com": 90.0, "tabloid.net": 20.0}
        result = compute_framing(articles, trust)
        assert result["framing_inconsistency"] < 0.05

    def test_orthogonal_framing_near_one(self, orthogonal_model):
        articles = (
            [_make_scored(f"h{i}", "bbc.com") for i in range(3)]
            + [_make_scored(f"l{i}", "tabloid.net") for i in range(3)]
        )
        trust = {"bbc.com": 90.0, "tabloid.net": 20.0}
        result = compute_framing(articles, trust)
        assert result["framing_inconsistency"] > 0.9

    def test_inconsistency_bounded(self, mixed_model):
        articles = (
            [_make_scored(f"h{i}", "bbc.com") for i in range(2)]
            + [_make_scored(f"l{i}", "tabloid.net") for i in range(2)]
        )
        trust = {"bbc.com": 90.0, "tabloid.net": 20.0}
        result = compute_framing(articles, trust)
        assert 0.0 <= result["framing_inconsistency"] <= 1.0

    def test_urls_partitioned_correctly(self, identical_model):
        high = [_make_scored(f"h{i}", "reuters.com", url=f"https://reuters.com/{i}") for i in range(2)]
        low = [_make_scored(f"l{i}", "fake.net", url=f"https://fake.net/{i}") for i in range(2)]
        trust = {"reuters.com": 85.0, "fake.net": 15.0}
        result = compute_framing(high + low, trust)
        assert set(result["high_trust_articles"]) == {a["url"] for a in high}
        assert set(result["low_trust_articles"]) == {a["url"] for a in low}

    def test_unknown_source_defaults_to_neutral(self, identical_model):
        # Unknown source defaults to trust=50, which is < 60 → low tier
        articles = (
            [_make_scored("h0", "bbc.com")]
            + [_make_scored("u0", "unknown-source.xyz")]
        )
        trust = {"bbc.com": 90.0}
        result = compute_framing(articles, trust)
        assert "https://example.com/u0" in result["low_trust_articles"]

    def test_result_has_all_keys(self, identical_model):
        articles = (
            [_make_scored("h0", "bbc.com")]
            + [_make_scored("l0", "tabloid.net")]
        )
        trust = {"bbc.com": 90.0, "tabloid.net": 10.0}
        result = compute_framing(articles, trust)
        for key in ("framing_inconsistency", "fact_inconsistency",
                    "high_trust_centroid", "low_trust_centroid",
                    "high_trust_articles", "low_trust_articles"):
            assert key in result

    def test_centroids_populated_when_both_tiers_present(self, mixed_model):
        articles = (
            [_make_scored(f"h{i}", "bbc.com") for i in range(2)]
            + [_make_scored(f"l{i}", "tabloid.net") for i in range(2)]
        )
        trust = {"bbc.com": 90.0, "tabloid.net": 20.0}
        result = compute_framing(articles, trust)
        assert len(result["high_trust_centroid"]) > 0
        assert len(result["low_trust_centroid"]) > 0

    def test_fact_inconsistency_zero_fallback_empty_articles(self):
        result = compute_framing([], {})
        assert result["fact_inconsistency"] == 0.0

    def test_fact_inconsistency_zero_fallback_one_tier(self):
        articles = [_make_scored(f"a{i}", "bbc.com") for i in range(3)]
        result = compute_framing(articles, {"bbc.com": 90.0})
        assert result["fact_inconsistency"] == 0.0


# ---------------------------------------------------------------------------
# _entity_overlap_score unit tests
# ---------------------------------------------------------------------------

def _fake_extract(responses: dict[str, dict]):
    """Return a mock extract_entities that maps cleaned_text → entity tags."""
    default = {"persons": [], "organizations": [], "locations": [], "events": []}
    def extract(text: str) -> dict:
        return responses.get(text, default)
    return extract


class TestEntityOverlapScore:
    def test_identical_entities_return_zero(self, monkeypatch):
        tags = {"persons": ["Alice"], "organizations": [], "locations": [], "events": []}
        monkeypatch.setattr(
            "src.scoring.framing.extract_entities",
            _fake_extract({"text_a": tags, "text_b": tags}),
        )
        high = [_make_scored("h0", "bbc.com", cleaned_text="text_a")]
        low = [_make_scored("l0", "fake.net", cleaned_text="text_b")]
        assert _entity_overlap_score(high, low) == 0.0

    def test_disjoint_entities_return_one(self, monkeypatch):
        monkeypatch.setattr(
            "src.scoring.framing.extract_entities",
            _fake_extract({
                "text_a": {"persons": ["Alice"], "organizations": [], "locations": [], "events": []},
                "text_b": {"persons": ["Bob"], "organizations": [], "locations": [], "events": []},
            }),
        )
        high = [_make_scored("h0", "bbc.com", cleaned_text="text_a")]
        low = [_make_scored("l0", "fake.net", cleaned_text="text_b")]
        assert _entity_overlap_score(high, low) == 1.0

    def test_partial_overlap_correct_value(self, monkeypatch):
        # high={alice, bob}, low={alice, carol} → |inter|=1, |union|=3 → 1 - 1/3 ≈ 0.6667
        monkeypatch.setattr(
            "src.scoring.framing.extract_entities",
            _fake_extract({
                "high_text": {"persons": ["Alice", "Bob"], "organizations": [], "locations": [], "events": []},
                "low_text": {"persons": ["Alice", "Carol"], "organizations": [], "locations": [], "events": []},
            }),
        )
        high = [_make_scored("h0", "bbc.com", cleaned_text="high_text")]
        low = [_make_scored("l0", "fake.net", cleaned_text="low_text")]
        result = _entity_overlap_score(high, low)
        assert abs(result - round(2 / 3, 4)) < 1e-4

    def test_no_entities_returns_zero(self, monkeypatch):
        monkeypatch.setattr(
            "src.scoring.framing.extract_entities",
            _fake_extract({}),  # all texts return empty default
        )
        high = [_make_scored("h0", "bbc.com", cleaned_text="text_a")]
        low = [_make_scored("l0", "fake.net", cleaned_text="text_b")]
        assert _entity_overlap_score(high, low) == 0.0

    def test_bounded_zero_to_one(self, monkeypatch):
        monkeypatch.setattr(
            "src.scoring.framing.extract_entities",
            _fake_extract({
                "text_a": {"persons": ["Alice", "Bob"], "organizations": ["UN"], "locations": ["Paris"], "events": []},
                "text_b": {"persons": ["Dave", "Eve"], "organizations": ["NATO"], "locations": ["Berlin"], "events": []},
            }),
        )
        high = [_make_scored("h0", "bbc.com", cleaned_text="text_a")]
        low = [_make_scored("l0", "fake.net", cleaned_text="text_b")]
        result = _entity_overlap_score(high, low)
        assert 0.0 <= result <= 1.0

    def test_orgs_and_locations_included(self, monkeypatch):
        # Verify ORG and location entities contribute to the overlap score
        monkeypatch.setattr(
            "src.scoring.framing.extract_entities",
            _fake_extract({
                "text_a": {"persons": [], "organizations": ["UN"], "locations": ["Paris"], "events": []},
                "text_b": {"persons": [], "organizations": ["UN"], "locations": ["Paris"], "events": []},
            }),
        )
        high = [_make_scored("h0", "bbc.com", cleaned_text="text_a")]
        low = [_make_scored("l0", "fake.net", cleaned_text="text_b")]
        assert _entity_overlap_score(high, low) == 0.0

    def test_score_is_float(self, monkeypatch):
        monkeypatch.setattr(
            "src.scoring.framing.extract_entities",
            _fake_extract({"t": {"persons": ["Alice"], "organizations": [], "locations": [], "events": []}}),
        )
        high = [_make_scored("h0", "bbc.com", cleaned_text="t")]
        low = [_make_scored("l0", "fake.net", cleaned_text="t")]
        assert isinstance(_entity_overlap_score(high, low), float)
