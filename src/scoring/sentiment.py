"""Per-article sentiment and sensationalism scoring (Week 2, Step 4)."""
from __future__ import annotations

from typing import TypedDict

from src.nlp.preprocessor import preprocess
from src.utils.models import RawItem

_pipeline = None

_SENSATIONAL_TERMS: frozenset[str] = frozenset({
    "shocking", "bombshell", "exposed", "explosive", "breaking",
    "urgent", "outrage", "scandal", "conspiracy", "hoax",
    "unbelievable", "stunning", "leaked", "alarming",
    "devastating", "catastrophic", "coverup", "cover-up",
})


class ScoredArticle(TypedDict):
    id: str
    title: str
    description: str | None
    source: str
    url: str
    platform: str
    timestamp: str
    engagement: dict[str, int]
    cleaned_text: str
    sentiment_label: str
    sentiment_score: float
    sentiment_extremity: float
    sensationalism_score: float


def _get_pipeline():
    """Lazy-load the HuggingFace sentiment pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            top_k=None,
        )
    return _pipeline


def _parse_scores(raw: list[dict]) -> tuple[str, float, float]:
    """Extract label, confidence, and extremity from a top_k=None pipeline result.

    Args:
        raw: List of dicts like [{"label": "positive", "score": 0.8}, ...].

    Returns:
        (label, score, extremity) where extremity = |positive_prob - negative_prob|.
    """
    scores = {entry["label"].lower(): entry["score"] for entry in raw}
    pos = scores.get("positive", 0.0)
    neg = scores.get("negative", 0.0)
    best = max(raw, key=lambda e: e["score"])
    return best["label"].lower(), best["score"], abs(pos - neg)


def _sensationalism(text: str) -> float:
    """Compute a sensationalism heuristic score in [0.0, 1.0].

    Combines ALL-CAPS word ratio (40%), exclamation density (30%),
    and loaded-term hits (30%).
    """
    words = text.split()
    if not words:
        return 0.0

    caps_ratio = sum(1 for w in words if len(w) > 1 and w.isupper()) / len(words)
    exclamation_score = min(text.count("!") / 3, 1.0)
    term_hits = sum(
        1 for w in words if w.lower().strip(".,!?;:'\"") in _SENSATIONAL_TERMS
    )
    term_score = min(term_hits / 3, 1.0)

    return round(caps_ratio * 0.4 + exclamation_score * 0.3 + term_score * 0.3, 4)


def score_article(item: RawItem) -> ScoredArticle:
    """Score a single article for sentiment and sensationalism.

    Args:
        item: A RawItem, typically from the NewsAPI scraper.

    Returns:
        ScoredArticle with cleaned_text, sentiment and sensationalism fields.
    """
    cleaned = preprocess(item)
    text = cleaned["cleaned_text"] or item["title"]

    pipe = _get_pipeline()
    label, score, extremity = _parse_scores(pipe([text])[0])

    return {
        **item,
        "cleaned_text": text,
        "sentiment_label": label,
        "sentiment_score": round(score, 4),
        "sentiment_extremity": round(extremity, 4),
        "sensationalism_score": _sensationalism(text),
    }  # type: ignore[return-value]


def score_articles(items: list[RawItem], batch_size: int = 16) -> list[ScoredArticle]:
    """Score a batch of articles for sentiment and sensationalism.

    Processes all texts in batched RoBERTa forward passes to avoid per-item
    model calls when scoring ~100 NewsAPI articles per topic.

    Args:
        items: Articles for a single topic cluster.
        batch_size: Texts per forward pass.

    Returns:
        ScoredArticles in the same order as input.
    """
    if not items:
        return []

    cleaned_items = [preprocess(item) for item in items]
    texts = [c["cleaned_text"] or item["title"] for c, item in zip(cleaned_items, items)]

    pipe = _get_pipeline()
    raw_results = pipe(texts, batch_size=batch_size)

    output: list[ScoredArticle] = []
    for item, text, raw in zip(items, texts, raw_results):
        label, score, extremity = _parse_scores(raw)
        output.append({
            **item,
            "cleaned_text": text,
            "sentiment_label": label,
            "sentiment_score": round(score, 4),
            "sentiment_extremity": round(extremity, 4),
            "sensationalism_score": _sensationalism(text),
        })  # type: ignore[misc]

    return output
