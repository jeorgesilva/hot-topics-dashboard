"""Framing consistency analysis across source trust tiers (Week 2, Step 5)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from src.scoring.sentiment import ScoredArticle

_model = None

_HIGH_TRUST_THRESHOLD: float = 60.0
_DEFAULT_TRUST: float = 50.0


class FramingResult(TypedDict):
    framing_inconsistency: float
    high_trust_centroid: list[float]
    low_trust_centroid: list[float]
    high_trust_articles: list[str]
    low_trust_articles: list[str]


def _get_model():
    """Lazy-load the sentence-transformers embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    centroid = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            centroid[i] += val
    n = len(vectors)
    return [x / n for x in centroid]


def compute_framing(
    articles: list[ScoredArticle],
    trust_scores: dict[str, float],
) -> FramingResult:
    """Compute framing inconsistency between high-trust and low-trust source tiers.

    Articles from sources with trust score >= 60 form the high-trust tier;
    the rest form the low-trust tier. Cosine distance between the mean
    embeddings of the two tiers measures how differently each group frames
    the topic. High divergence is a misinformation signal.

    If either tier is empty (e.g. all sources are credible), framing_inconsistency
    falls back to 0.0 so the composite scorer is not penalised for missing data.

    Args:
        articles: Scored articles for a single topic (output of score_articles).
        trust_scores: Source name -> trust score (0–100). Sources absent from
            the map default to 50 (neutral).

    Returns:
        FramingResult with framing_inconsistency in [0.0, 1.0] and centroid
        vectors stored for the Week 3 dashboard explainability panel.
    """
    if not articles:
        return FramingResult(
            framing_inconsistency=0.0,
            high_trust_centroid=[],
            low_trust_centroid=[],
            high_trust_articles=[],
            low_trust_articles=[],
        )

    high_articles = [
        a for a in articles
        if trust_scores.get(a["source"], _DEFAULT_TRUST) >= _HIGH_TRUST_THRESHOLD
    ]
    low_articles = [
        a for a in articles
        if trust_scores.get(a["source"], _DEFAULT_TRUST) < _HIGH_TRUST_THRESHOLD
    ]

    if not high_articles or not low_articles:
        return FramingResult(
            framing_inconsistency=0.0,
            high_trust_centroid=[],
            low_trust_centroid=[],
            high_trust_articles=[a["url"] for a in high_articles],
            low_trust_articles=[a["url"] for a in low_articles],
        )

    model = _get_model()

    high_texts = [a["cleaned_text"] or a["title"] for a in high_articles]
    low_texts = [a["cleaned_text"] or a["title"] for a in low_articles]

    high_embeddings: list[list[float]] = model.encode(high_texts).tolist()
    low_embeddings: list[list[float]] = model.encode(low_texts).tolist()

    high_centroid = _mean_vector(high_embeddings)
    low_centroid = _mean_vector(low_embeddings)

    similarity = _cosine_similarity(high_centroid, low_centroid)
    inconsistency = max(0.0, min(1.0, round(1.0 - similarity, 4)))

    return FramingResult(
        framing_inconsistency=inconsistency,
        high_trust_centroid=high_centroid,
        low_trust_centroid=low_centroid,
        high_trust_articles=[a["url"] for a in high_articles],
        low_trust_articles=[a["url"] for a in low_articles],
    )
