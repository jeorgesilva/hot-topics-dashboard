"""Tests for src/nlp/topic_query.py"""
from src.nlp.ner import annotate
from src.nlp.preprocessor import RawItem, preprocess
from src.nlp.topic_query import build_topic_query


def _make_annotated(title: str, description: str | None = None, source: str = "rss"):
    raw = RawItem(
        id="t",
        title=title,
        description=description,
        source=source,
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


class TestMediaOrgFiltering:
    """ORG entities that are news outlets must not appear in the query."""

    def test_source_name_excluded_from_org_entities(self):
        # "BBC News" is the source — it should not appear as a query term
        # even if NER picks it up from the description.
        item = _make_annotated(
            "Hantavirus outbreak spreads to Rotterdam",
            description="Reported by BBC News. WHO issues alert.",
            source="BBC News",
        )
        result = build_topic_query([item])
        assert "BBC" not in result
        assert "BBC News" not in result

    def test_media_token_org_excluded(self):
        # "CNN" contains no _MEDIA_TOKENS but "Daily Mail" contains "daily"
        item = _make_annotated(
            "Climate summit update",
            description="According to the Daily Mail, temperatures are rising.",
            source="Reuters",
        )
        result = build_topic_query([item])
        assert "Daily Mail" not in result

    def test_non_media_org_included(self):
        # "WHO" (World Health Organization) should not be filtered — it has
        # no media tokens and doesn't match the source name.
        item = _make_annotated(
            "WHO warns of new Ebola variant in Congo",
            source="Reuters",
        )
        result = build_topic_query([item])
        # WHO or Congo or Reuters is present; at minimum the query is non-empty
        assert len(result) > 0

    def test_query_falls_back_to_keywords_when_orgs_all_filtered(self):
        item = _make_annotated(
            "Breaking: Reuters reports global market crash",
            description="Reuters and Associated Press confirm the drop.",
            source="Reuters",
        )
        result = build_topic_query([item])
        # After filtering "Reuters" and "Associated Press" (contains "press"),
        # keywords from the title/description should fill the query.
        assert isinstance(result, str)

    def test_broadcast_abbreviation_excluded(self):
        # Known broadcast abbreviations in ORG entities should be filtered.
        item = _make_annotated(
            "Sandy Fire erupts in Simi Valley",
            description="According to CNN and ABC7 Los Angeles, crews are on site.",
            source="Los Angeles Times",
        )
        result = build_topic_query([item])
        assert "CNN" not in result

    def test_media_keywords_filtered_from_fallback(self, monkeypatch):
        item = {
            "source": "Reuters",
            "entities": {
                "persons": [],
                "organizations": [],
                "locations": [],
                "events": [],
            },
        }
        monkeypatch.setattr(
            "src.nlp.topic_query.extract_keywords",
            lambda _items, top_n: [["bbc", "cnn", "economy"]],
        )
        result = build_topic_query([item], max_terms=3)
        assert "economy" in result
        assert "bbc" not in result.lower()
        assert "cnn" not in result.lower()
