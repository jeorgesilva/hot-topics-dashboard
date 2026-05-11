"""Tests for src/nlp/keywords.py"""
from src.nlp.keywords import attach_keywords, extract_keywords
from src.nlp.preprocessor import preprocess_batch


class TestExtractKeywords:
    def test_returns_one_list_per_item(self, mock_raw_items):
        cleaned = preprocess_batch(mock_raw_items)
        results = extract_keywords(cleaned)
        assert len(results) == len(cleaned)

    def test_keywords_are_strings(self, mock_raw_items):
        cleaned = preprocess_batch(mock_raw_items)
        results = extract_keywords(cleaned)
        for kw_list in results:
            for kw in kw_list:
                assert isinstance(kw, str)

    def test_top_n_respected(self, mock_raw_items):
        cleaned = preprocess_batch(mock_raw_items)
        results = extract_keywords(cleaned, top_n=3)
        for kw_list in results:
            assert len(kw_list) <= 3

    def test_empty_corpus(self):
        assert extract_keywords([]) == []


class TestAttachKeywords:
    def test_adds_keywords_field(self, mock_raw_items):
        cleaned = preprocess_batch(mock_raw_items)
        result = attach_keywords(cleaned)
        for item in result:
            assert "keywords" in item

    def test_returns_same_items(self, mock_raw_items):
        cleaned = preprocess_batch(mock_raw_items)
        result = attach_keywords(cleaned)
        assert len(result) == len(cleaned)
