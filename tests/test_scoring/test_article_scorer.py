"""Tests for src/scoring/article_scorer.py."""
from __future__ import annotations

import math

import pytest

from src.scoring.article_scorer import score_article


def test_low_risk_article():
    """Trusted source, neutral tone, no sensationalism → score < 0.2."""
    score = score_article(
        trust_score=90,
        sentiment_extremity=0.05,
        sensationalism_score=0.05,
        attribution_vagueness=0.0,
    )
    assert 0.0 <= score <= 1.0
    assert score < 0.2, f"Expected low-risk score < 0.2, got {score}"
    assert not math.isnan(score)


def test_high_risk_article():
    """Untrustworthy source, extreme sentiment, high sensationalism → score > 0.7."""
    score = score_article(
        trust_score=15,
        sentiment_extremity=0.90,
        sensationalism_score=0.85,
        attribution_vagueness=0.80,
    )
    assert 0.0 <= score <= 1.0
    assert score > 0.7, f"Expected high-risk score > 0.7, got {score}"


def test_input_clamping():
    """Out-of-range inputs must be clamped silently; result stays in [0, 1]."""
    score = score_article(
        trust_score=110,      # above 100
        sentiment_extremity=1.5,  # above 1
        sensationalism_score=-0.1,  # below 0
        attribution_vagueness=0.5,
    )
    assert 0.0 <= score <= 1.0, f"Clamped score out of range: {score}"
    assert not math.isnan(score)


def test_formula_weights():
    """Verify the weighted formula produces the expected value."""
    trust_score = 60.0
    sentiment_extremity = 0.4
    sensationalism_score = 0.3
    attribution_vagueness = 0.2

    expected = round(
        0.15 * (1.0 - trust_score / 100.0)
        + 0.30 * sentiment_extremity
        + 0.30 * sensationalism_score
        + 0.25 * attribution_vagueness,
        4,
    )
    result = score_article(
        trust_score=trust_score,
        sentiment_extremity=sentiment_extremity,
        sensationalism_score=sensationalism_score,
        attribution_vagueness=attribution_vagueness,
    )
    assert result == pytest.approx(expected, abs=1e-4)


def test_full_trust_zero_signals():
    """Perfect source (trust=100) with zero signal → purely distrust-driven (0)."""
    score = score_article(
        trust_score=100,
        sentiment_extremity=0.0,
        sensationalism_score=0.0,
        attribution_vagueness=0.0,
    )
    assert score == pytest.approx(0.0, abs=1e-6)


def test_zero_trust_max_signals():
    """Zero trust and all signals maxed → score approaches 1.0."""
    score = score_article(
        trust_score=0,
        sentiment_extremity=1.0,
        sensationalism_score=1.0,
        attribution_vagueness=1.0,
    )
    assert score == pytest.approx(1.0, abs=1e-6)
