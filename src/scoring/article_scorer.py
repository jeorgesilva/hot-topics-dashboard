"""Per-article risk score combining four normalized misinformation signals."""
from __future__ import annotations


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

    Formula:
        source_distrust = 1.0 - clamp(trust_score, 0, 100) / 100.0
        risk = 0.15 × source_distrust
             + 0.30 × sentiment_extremity
             + 0.30 × sensationalism_score
             + 0.25 × attribution_vagueness
    """
    trust_clamped = max(0.0, min(100.0, float(trust_score)))
    sent_clamped  = max(0.0, min(1.0, float(sentiment_extremity)))
    sens_clamped  = max(0.0, min(1.0, float(sensationalism_score)))
    attr_clamped  = max(0.0, min(1.0, float(attribution_vagueness)))

    source_distrust = 1.0 - trust_clamped / 100.0
    return round(
        0.15 * source_distrust
        + 0.30 * sent_clamped
        + 0.30 * sens_clamped
        + 0.25 * attr_clamped,
        4,
    )
