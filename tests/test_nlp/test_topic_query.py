"""Tests for src/nlp/topic_query.py"""
import pytest

from src.nlp.ner import annotate
from src.nlp.preprocessor import RawItem, preprocess
from src.nlp.topic_query import build_topic_query


def _make_annotated(title: str, description: str | None = None):
    raw = RawItem(
        id="t",
        title=title,
        description=description,
        source="rss",
        url="https://example.com",
        platform="rss",
        timestamp="2026-05-15T00:00:00Z",
        engagement={},
    )
    return annotate(preprocess(raw))


class TestBuildTopicQuery:
    def test_returns_string(self):
        item = _make_annotated("EU proposes new AI regulation framework in Brussels")
        result = build_topic_query([item])
        assert isinstance(result, str)

    def test_empty_list_returns_empty_string(self):
        assert build_topic_query([]) == ""

    def test_respects_max_terms(self):
        item = _make_annotated(
            "Elon Musk Tesla SpaceX Paris Brussels Berlin Rome London Tokyo"
        )
        result = build_topic_query([item], max_terms=3)
        assert len(result.split()) <= 3

    def test_no_duplicate_terms(self):
        item = _make_annotated(
            "NASA NASA NASA launched a rocket from Florida",
            description="NASA announced the launch from Kennedy Space Center in Florida.",
        )
        result = build_topic_query([item])
        words = [w.lower() for w in result.split()]
        assert len(words) == len(set(words))

    def test_ner_entities_appear_in_query(self):
        item = _make_annotated("Emmanuel Macron visits Berlin for EU summit")
        result = build_topic_query([item])
        assert any(term in result for term in ["Macron", "Berlin", "EU", "Emmanuel"])

    def test_multiple_items_merged(self):
        items = [
            _make_annotated("EU AI regulation vote in Brussels"),
            _make_annotated("European Parliament debates AI Act framework"),
        ]
        result = build_topic_query(items)
        assert len(result) > 0

    def test_single_char_terms_excluded(self):
        item = _make_annotated("A B C D E F G H I regulation framework")
        result = build_topic_query([item])
        for term in result.split():
            assert len(term) > 1
