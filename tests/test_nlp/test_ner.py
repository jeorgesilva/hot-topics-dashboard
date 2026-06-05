"""Tests for src/nlp/ner.py"""
from src.nlp.ner import annotate, annotate_batch, extract_entities
from src.nlp.preprocessor import preprocess


class TestExtractEntities:
    def test_returns_entity_tags_structure(self):
        tags = extract_entities("Angela Merkel besuchte Berlin.")
        assert "persons" in tags
        assert "organizations" in tags
        assert "locations" in tags
        assert "events" in tags

    def test_detects_person(self):
        tags = extract_entities("Angela Merkel traf sich mit Olaf Scholz in Berlin.")
        assert any("Merkel" in p or "Angela" in p for p in tags["persons"])

    def test_detects_location(self):
        tags = extract_entities("Der Gipfel fand in Berlin statt.")
        locations = [loc.lower() for loc in tags["locations"]]
        assert "berlin" in locations

    def test_no_duplicates(self):
        tags = extract_entities("Die Bundesregierung Bundesregierung startete eine Initiative.")
        for key in tags:
            assert len(tags[key]) == len(set(t.lower() for t in tags[key]))

    def test_empty_text(self):
        tags = extract_entities("")
        assert tags == {"persons": [], "organizations": [], "locations": [], "events": []}


class TestAnnotate:
    def test_adds_entities_field(self, mock_raw_item):
        cleaned = preprocess(mock_raw_item)
        annotated = annotate(cleaned)
        assert "entities" in annotated

    def test_preserves_cleaned_fields(self, mock_raw_item):
        cleaned = preprocess(mock_raw_item)
        annotated = annotate(cleaned)
        assert annotated["cleaned_text"] == cleaned["cleaned_text"]
        assert annotated["id"] == cleaned["id"]


class TestAnnotateBatch:
    def test_returns_same_length(self, mock_raw_items):
        cleaned_items = [preprocess(item) for item in mock_raw_items]
        results = annotate_batch(cleaned_items)
        assert len(results) == len(cleaned_items)

    def test_empty_list(self):
        assert annotate_batch([]) == []
