"""Framing consistency analysis across source trust tiers (Week 2, Step 5)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from src.nlp.ner import extract_entities

if TYPE_CHECKING:
    from src.scoring.sentiment import ScoredArticle

_model = None

_HIGH_TRUST_THRESHOLD: float = 60.0
_DEFAULT_TRUST: float = 50.0


class FramingResult(TypedDict):
    framing_inconsistency: float
    fact_inconsistency: float
    high_trust_centroid: list[float]
    low_trust_centroid: list[float]
    high_trust_articles: list[str]
    low_trust_articles: list[str]


def _get_model():
    """Lazy-load the sentence-transformers embedding model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
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


def _intra_cluster_variance(articles: list[ScoredArticle], model) -> float:
    """Mean cosine distance from centroid — framing diversity within a single tier.

    Used as a fallback when all articles belong to one trust tier and the
    cross-tier inconsistency signal cannot be computed. A high variance means
    even the homogeneous tier frames the topic in divergent ways, which is a
    weaker but still informative misinformation signal.

    Args:
        articles: Articles in the single available tier.
        model: Loaded SentenceTransformer instance.

    Returns:
        Mean cosine distance from centroid in [0.0, 1.0].
    """
    texts = [a["cleaned_text"] or a["title"] for a in articles]
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return 0.0
    embeddings: list[list[float]] = model.encode(texts, batch_size=32, show_progress_bar=False).tolist()
    centroid = _mean_vector(embeddings)
    distances = [max(0.0, 1.0 - _cosine_similarity(emb, centroid)) for emb in embeddings]
    return round(sum(distances) / len(distances), 4)


def _entity_overlap_score(
    high_articles: list[ScoredArticle],
    low_articles: list[ScoredArticle],
) -> float:
    """Jaccard distance between named-entity sets of the two trust tiers.

    Extracts PERSON, ORG, and location entities from each tier's cleaned
    text and returns 1 - (|intersection| / |union|). Returns 0.0 when
    neither tier yields extractable entities to avoid penalising topics
    where spaCy NER produces no annotations.

    Args:
        high_articles: Articles in the high-trust tier.
        low_articles: Articles in the low-trust tier.

    Returns:
        Entity inconsistency in [0.0, 1.0].
    """
    def _entity_set(articles: list[ScoredArticle]) -> set[str]:
        entities: set[str] = set()
        for a in articles:
            tags = extract_entities(a["cleaned_text"] or a["title"])
            for bucket in ("persons", "organizations", "locations"):
                for e in tags[bucket]:
                    entities.add(e.lower().strip())
        return entities

    high_ents = _entity_set(high_articles)
    low_ents = _entity_set(low_articles)
    union = high_ents | low_ents
    if not union:
        return 0.0
    return round(1.0 - len(high_ents & low_ents) / len(union), 4)


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
            fact_inconsistency=0.0,
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
        # Only one trust tier present — use intra-cluster embedding variance
        # as a proxy for framing diversity instead of returning a flat 0.0.
        single_tier = high_articles or low_articles
        if single_tier:
            variance = _intra_cluster_variance(single_tier, _get_model())
        else:
            variance = 0.0
        return FramingResult(
            framing_inconsistency=variance,
            fact_inconsistency=0.0,
            high_trust_centroid=[],
            low_trust_centroid=[],
            high_trust_articles=[a["url"] for a in high_articles],
            low_trust_articles=[a["url"] for a in low_articles],
        )

    model = _get_model()

    high_texts = [
        a["cleaned_text"] or a["title"] for a in high_articles
        if a.get("cleaned_text") or a.get("title")
    ]
    low_texts = [
        a["cleaned_text"] or a["title"] for a in low_articles
        if a.get("cleaned_text") or a.get("title")
    ]

    if not high_texts or not low_texts:
        single_tier = high_articles or low_articles
        variance = _intra_cluster_variance(single_tier, model) if single_tier else 0.0
        return FramingResult(
            framing_inconsistency=variance,
            fact_inconsistency=0.0,
            high_trust_centroid=[],
            low_trust_centroid=[],
            high_trust_articles=[a["url"] for a in high_articles],
            low_trust_articles=[a["url"] for a in low_articles],
        )

    high_embeddings = model.encode(high_texts, batch_size=32, show_progress_bar=False)
    low_embeddings = model.encode(low_texts, batch_size=32, show_progress_bar=False)

    high_centroid: list[float] = np.mean(high_embeddings, axis=0).tolist()
    low_centroid: list[float] = np.mean(low_embeddings, axis=0).tolist()

    similarity = _cosine_similarity(high_centroid, low_centroid)
    inconsistency = max(0.0, min(1.0, round(1.0 - similarity, 4)))

    return FramingResult(
        framing_inconsistency=inconsistency,
        fact_inconsistency=_entity_overlap_score(high_articles, low_articles),
        high_trust_centroid=high_centroid,
        low_trust_centroid=low_centroid,
        high_trust_articles=[a["url"] for a in high_articles],
        low_trust_articles=[a["url"] for a in low_articles],
    )
