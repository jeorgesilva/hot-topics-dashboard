"""Per-article risk score combining four normalized misinformation signals."""
from __future__ import annotations

from src.scoring.weights import ARTICLE_RISK_WEIGHTS


def score_article(
    trust_score: float,
    sentiment_extremity: float,
    sensationalism_score: float,
    attribution_vagueness: float,
) -> float:
    """Calculate article_risk_score combining 4 normalised signals.

    Args:
        trust_score: Source trust in [0, 100] from domain_resolver / MBFC.
        sentiment_extremity: |positive_prob - negative_prob| in [0, 1].
        sensationalism_score: Composite caps + exclamation + clickbait score in [0, 1].
        attribution_vagueness: Density of vague attribution patterns in [0, 1].

    Returns:
        float in [0.0, 1.0]. Higher = higher individual article risk.

    Formula (weights from src/scoring/weights.py::ARTICLE_RISK_WEIGHTS):
        source_distrust = 1.0 - clamp(trust_score, 0, 100) / 100.0
        risk = source_distrust * ARTICLE_RISK_WEIGHTS["source_distrust"]
             + sentiment_extremity * ARTICLE_RISK_WEIGHTS["sentiment_extremity"]
             + sensationalism_score * ARTICLE_RISK_WEIGHTS["sensationalism"]
             + attribution_vagueness * ARTICLE_RISK_WEIGHTS["attribution_vagueness"]
    """
    trust_clamped = max(0.0, min(100.0, float(trust_score)))
    sent_clamped  = max(0.0, min(1.0, float(sentiment_extremity)))
    sens_clamped  = max(0.0, min(1.0, float(sensationalism_score)))
    attr_clamped  = max(0.0, min(1.0, float(attribution_vagueness)))

    source_distrust = 1.0 - trust_clamped / 100.0
    return round(
        ARTICLE_RISK_WEIGHTS["source_distrust"] * source_distrust
        + ARTICLE_RISK_WEIGHTS["sentiment_extremity"] * sent_clamped
        + ARTICLE_RISK_WEIGHTS["sensationalism"] * sens_clamped
        + ARTICLE_RISK_WEIGHTS["attribution_vagueness"] * attr_clamped,
        4,
    )
