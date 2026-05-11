"""Tests for src/nlp/preprocessor.py"""
import pytest

from src.nlp.preprocessor import (
    CleanedItem,
    clean_text,
    normalize_unicode,
    preprocess,
    preprocess_batch,
    strip_html,
    tokenize_and_lemmatize,
)


class TestStripHtml:
    def test_removes_tags(self):
        assert strip_html("<b>bold</b> text") == " bold  text"

    def test_no_html_unchanged(self):
        assert strip_html("plain text") == "plain text"

    def test_nested_tags(self):
        result = strip_html("<div><p>hello</p></div>")
        assert "hello" in result
        assert "<" not in result


class TestNormalizeUnicode:
    def test_nfc_normalization(self):
        # café composed vs decomposed form
        decomposed = "café"
        composed = "café"
        assert normalize_unicode(decomposed) == composed

    def test_strips_control_characters(self):
        assert "\x00" not in normalize_unicode("hello\x00world")


class TestCleanText:
    def test_strips_html_and_collapses_whitespace(self, mock_raw_item):
        result = clean_text(mock_raw_item["title"])
        assert "<b>" not in result
        assert "</b>" not in result
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_whitespace(self):
        assert clean_text("   ") == ""


class TestTokenizeAndLemmatize:
    def test_returns_two_lists(self):
        tokens, lemmas = tokenize_and_lemmatize("Scientists discover new variant")
        assert isinstance(tokens, list)
        assert isinstance(lemmas, list)

    def test_removes_stop_words(self):
        tokens, _ = tokenize_and_lemmatize("the scientists and the doctors")
        assert "the" not in tokens
        assert "and" not in tokens

    def test_lemmatization_applied(self):
        _, lemmas = tokenize_and_lemmatize("scientists are discovering variants")
        assert "scientist" in lemmas or "discover" in lemmas

    def test_non_alpha_tokens_excluded(self):
        tokens, _ = tokenize_and_lemmatize("hello 123 world !!!")
        for t in tokens:
            assert t.isalpha()


class TestPreprocess:
    def test_output_has_required_fields(self, mock_raw_item):
        result = preprocess(mock_raw_item)
        assert "cleaned_text" in result
        assert "tokens" in result
        assert "lemmas" in result

    def test_preserves_original_fields(self, mock_raw_item):
        result = preprocess(mock_raw_item)
        assert result["id"] == mock_raw_item["id"]
        assert result["platform"] == mock_raw_item["platform"]

    def test_combines_title_and_description(self, mock_raw_item):
        result = preprocess(mock_raw_item)
        assert "Health" in result["cleaned_text"] or "health" in result["cleaned_text"]

    def test_handles_none_description(self, mock_raw_items):
        item_no_desc = mock_raw_items[2]  # description=None
        result = preprocess(item_no_desc)
        assert isinstance(result["cleaned_text"], str)
        assert len(result["cleaned_text"]) > 0

    def test_html_stripped_from_title(self, mock_raw_item):
        result = preprocess(mock_raw_item)
        assert "<b>" not in result["cleaned_text"]


class TestPreprocessBatch:
    def test_returns_same_length(self, mock_raw_items):
        results = preprocess_batch(mock_raw_items)
        assert len(results) == len(mock_raw_items)

    def test_empty_list(self):
        assert preprocess_batch([]) == []

    def test_all_items_have_cleaned_text(self, mock_raw_items):
        results = preprocess_batch(mock_raw_items)
        for r in results:
            assert isinstance(r["cleaned_text"], str)
