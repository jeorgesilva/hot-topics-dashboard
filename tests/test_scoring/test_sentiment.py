"""Tests for src/scoring/sentiment.py — HuggingFace pipeline is always mocked."""
from __future__ import annotations

import pytest

from src.scoring.sentiment import _parse_scores, _sensationalism, score_article, score_articles
from src.utils.models import RawItem


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_item(
    item_id: str = "newsapi_001",
    title: str = "Scientists announce major cancer treatment breakthrough",
    description: str | None = "New therapy shows 90% success rate in trials.",
    source: str = "bbc.com",
) -> RawItem:
    return RawItem(
        id=item_id,
        title=title,
        description=description,
        source=source,
        url=f"https://bbc.com/news/{item_id}",
        platform="newsapi",
        timestamp="2026-05-11T10:00:00Z",
        engagement={"score": 0, "comments": 0},
    )


def _fake_pipeline(label: str = "positive", pos: float = 0.75, neg: float = 0.05):
    """Return a fake pipeline callable with fixed scores."""
    neutral = round(1.0 - pos - neg, 4)

    def predict(texts, batch_size=16):
        return [
            [
                {"label": "positive", "score": pos},
                {"label": "neutral", "score": neutral},
                {"label": "negative", "score": neg},
            ]
            for _ in texts
        ]

    return predict


@pytest.fixture
def mock_positive_pipeline(monkeypatch):
    monkeypatch.setattr("src.scoring.sentiment._get_pipeline", lambda: _fake_pipeline("positive", 0.75, 0.05))


@pytest.fixture
def mock_neutral_pipeline(monkeypatch):
    monkeypatch.setattr("src.scoring.sentiment._get_pipeline", lambda: _fake_pipeline("neutral", 0.33, 0.34))


@pytest.fixture
def mock_extreme_pipeline(monkeypatch):
    # Maximally extreme: positive=0.99, negative=0.005
    monkeypatch.setattr("src.scoring.sentiment._get_pipeline", lambda: _fake_pipeline("positive", 0.99, 0.005))


# ---------------------------------------------------------------------------
# _parse_scores unit tests
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_returns_dominant_label(self):
        raw = [
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "negative", "score": 0.05},
        ]
        label, score, extremity = _parse_scores(raw)
        assert label == "positive"
        assert score == 0.8

    def test_extremity_is_pos_minus_neg(self):
        raw = [
            {"label": "positive", "score": 0.8},
            {"label": "neutral", "score": 0.15},
            {"label": "negative", "score": 0.05},
        ]
        _, _, extremity = _parse_scores(raw)
        assert abs(extremity - 0.75) < 1e-6

    def test_neutral_article_low_extremity(self):
        raw = [
            {"label": "positive", "score": 0.33},
            {"label": "neutral", "score": 0.34},
            {"label": "negative", "score": 0.33},
        ]
        _, _, extremity = _parse_scores(raw)
        assert extremity < 0.05

    def test_label_case_insensitive(self):
        raw = [
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEUTRAL", "score": 0.05},
            {"label": "NEGATIVE", "score": 0.05},
        ]
        label, _, _ = _parse_scores(raw)
        assert label == "positive"


# ---------------------------------------------------------------------------
# _sensationalism unit tests
# ---------------------------------------------------------------------------

class TestSensationalism:
    def test_empty_text_returns_zero(self):
        assert _sensationalism("") == 0.0

    def test_plain_text_low_score(self):
        score = _sensationalism("Scientists publish research on climate change adaptation.")
        assert score < 0.1

    def test_all_caps_increases_score(self):
        score = _sensationalism("BREAKING NEWS ALERT MAJOR EVENT HAPPENING NOW")
        assert score > 0.3

    def test_exclamations_increase_score(self):
        score = _sensationalism("This is unbelievable!!! You won't believe it!!!")
        assert score > 0.2

    def test_loaded_terms_increase_score(self):
        score = _sensationalism("shocking bombshell exposed conspiracy hoax leaked")
        assert score > 0.2

    def test_score_bounded_zero_to_one(self):
        extreme = "SHOCKING!!! BOMBSHELL!!! EXPOSED!!! CONSPIRACY HOAX LEAKED SCANDAL!!!"
        score = _sensationalism(extreme)
        assert 0.0 <= score <= 1.0

    def test_score_is_float(self):
        assert isinstance(_sensationalism("some text"), float)

    def test_short_acronyms_dont_inflate_caps(self):
        # WHO, FBI, CIA, EU, UK are legitimate acronyms (len ≤ 3). They should
        # not trigger the caps signal the way SHOCKING or BOMBSHELL would.
        legitimate = "WHO warns that FBI and CIA have EU and UK concerns"
        sensational = "SHOCKING BOMBSHELL EXPOSED conspiracy LEAKED"
        assert _sensationalism(legitimate) < _sensationalism(sensational)

    def test_four_char_acronym_still_counted(self):
        # Words with len > 3 (e.g. NASA, CNBC) are still assessed.
        score = _sensationalism("NASA CNBC ESPN MSNBC confirms report")
        # These are caps words of len 4+ so they DO contribute, but the
        # sentence has no exclamations or loaded terms — score stays moderate.
        assert 0.0 < score < 0.5


# ---------------------------------------------------------------------------
# score_article integration tests
# ---------------------------------------------------------------------------

class TestScoreArticle:
    def test_returns_all_required_fields(self, mock_positive_pipeline):
        result = score_article(_make_item())
        for field in (
            "id", "title", "description", "source", "url", "platform",
            "timestamp", "engagement", "cleaned_text",
            "sentiment_label", "sentiment_score", "sentiment_extremity",
            "sensationalism_score",
        ):
            assert field in result, f"missing field: {field}"

    def test_passthrough_fields_unchanged(self, mock_positive_pipeline):
        item = _make_item()
        result = score_article(item)
        assert result["id"] == item["id"]
        assert result["source"] == item["source"]
        assert result["url"] == item["url"]

    def test_cleaned_text_populated(self, mock_positive_pipeline):
        result = score_article(_make_item())
        assert isinstance(result["cleaned_text"], str)
        assert len(result["cleaned_text"]) > 0

    def test_sentiment_scores_in_range(self, mock_positive_pipeline):
        result = score_article(_make_item())
        assert 0.0 <= result["sentiment_score"] <= 1.0
        assert 0.0 <= result["sentiment_extremity"] <= 1.0
        assert 0.0 <= result["sensationalism_score"] <= 1.0

    def test_neutral_article_low_extremity(self, mock_neutral_pipeline):
        result = score_article(_make_item())
        assert result["sentiment_extremity"] < 0.1

    def test_extreme_article_high_extremity(self, mock_extreme_pipeline):
        result = score_article(_make_item())
        assert result["sentiment_extremity"] > 0.9

    def test_none_description_handled(self, mock_positive_pipeline):
        item = _make_item(description=None)
        result = score_article(item)
        assert result["cleaned_text"] != ""


# ---------------------------------------------------------------------------
# score_articles batch tests
# ---------------------------------------------------------------------------

class TestScoreArticles:
    def test_empty_list_returns_empty(self):
        assert score_articles([]) == []

    def test_returns_same_length(self, mock_positive_pipeline):
        items = [_make_item(item_id=f"newsapi_{i:03d}") for i in range(5)]
        results = score_articles(items)
        assert len(results) == 5

    def test_all_scores_in_range(self, mock_positive_pipeline):
        items = [_make_item(item_id=f"newsapi_{i:03d}") for i in range(10)]
        results = score_articles(items)
        for r in results:
            assert 0.0 <= r["sentiment_score"] <= 1.0
            assert 0.0 <= r["sentiment_extremity"] <= 1.0
            assert 0.0 <= r["sensationalism_score"] <= 1.0

    def test_order_preserved(self, mock_positive_pipeline):
        items = [_make_item(item_id=f"newsapi_{i:03d}") for i in range(3)]
        results = score_articles(items)
        for item, result in zip(items, results):
            assert result["id"] == item["id"]

    def test_sensational_item_scores_higher(self, mock_positive_pipeline):
        normal = _make_item(item_id="n001", title="Scientists publish new research findings.")
        sensational = _make_item(
            item_id="s001",
            title="SHOCKING BOMBSHELL!!! Government EXPOSED!!!",
            description="The TRUTH they don't want you to know. Conspiracy LEAKED!!!",
        )
        results = score_articles([normal, sensational])
        assert results[1]["sensationalism_score"] > results[0]["sensationalism_score"]
