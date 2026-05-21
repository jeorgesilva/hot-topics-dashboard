"""Per-article sentiment and sensationalism scoring (Week 2, Step 4)."""
from __future__ import annotations

import re
from typing import TypedDict

from src.nlp.preprocessor import preprocess
from src.utils.models import RawItem

_pipeline = None

# ── Sensationalism lexicon ────────────────────────────────────────────────────
# Expanded from 18 to ~80 single-word terms grouped by semantic category.
# Multi-word patterns are handled separately by _CLICKBAIT_PATTERNS.
_SENSATIONAL_TERMS: frozenset[str] = frozenset({
    # Urgency / Breaking
    "breaking", "urgent", "alert", "developing", "exclusive",
    "emergency", "flash", "live",
    # Conspiracy / Cover-up
    "conspiracy", "coverup", "cover-up", "censored", "suppressed",
    "whistleblower", "plandemic", "scamdemic", "globalist", "banned",
    "shadowbanned", "deepstate",
    # Revelation / Exposure
    "exposed", "leaked", "revealed", "uncovered", "bombshell", "explosive",
    "damning", "incriminating", "scandalous",
    # Fear / Catastrophe
    "catastrophic", "devastating", "alarming", "terrifying", "horrifying",
    "crisis", "collapse", "apocalyptic", "existential", "annihilation",
    "extinction",
    # Outrage / Scandal
    "outrage", "scandal", "shocking", "disgrace", "disgusting",
    "appalling", "unbelievable", "stunning", "staggering", "outrageous",
    "unthinkable", "despicable",
    # Medical misinformation
    "hoax", "depopulation", "miracle",
    # Political sensationalism
    "treason", "traitor", "traitors", "coup", "rigged", "stolen",
    "overthrow",
    # Deception
    "lied", "lying", "liar", "liars", "deceived", "deceiving",
    "fabricated", "falsified", "manipulation",
    # Superlatives / Hyperbole
    "unprecedented", "unimaginable", "mind-blowing", "earth-shattering",
    "jaw-dropping",
})

# ── Structural clickbait patterns ─────────────────────────────────────────────
# Regex patterns that detect manipulation framings independent of individual
# word choices. Compiled once at import time for performance.
_CLICKBAIT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"you won'?t believe", re.IGNORECASE),
    re.compile(r"the (real|true|hidden|secret|shocking) (reason|truth|story)", re.IGNORECASE),
    re.compile(r"(they|he|she|doctors?|scientists?|experts?|the government).{0,20}don'?t want you", re.IGNORECASE),
    re.compile(r"is .{3,40}(hiding|lying|corrupt\b|guilty\b)", re.IGNORECASE),
    re.compile(r"\d+\s+(things?|reasons?|ways?|facts?|secrets?).{0,25}(shocking|amaz|unbeliev|crazy|wild)", re.IGNORECASE),
    re.compile(r"(finally|now|just)\s+(revealed|exposed|confirmed|proven|admitted)", re.IGNORECASE),
    re.compile(r"(mainstream media|msm|fake news).{0,30}(lies?|hiding|won'?t|ignor)", re.IGNORECASE),
    re.compile(r"what (they|the media|scientists?|doctors?|experts?) (don'?t|won'?t|refuse to)", re.IGNORECASE),
)


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


def _clickbait_score(text: str) -> float:
    """Return a structural clickbait score in [0.0, 1.0].

    Detects manipulation framings that bypass single-word lexicons:
    withholding constructions ("they don't want you to know"), rhetorical
    accusations ("is X hiding?"), list-bait ("5 things that will shock you"),
    and false-revelation structures ("finally revealed").
    Three or more pattern matches saturates the score at 1.0.
    """
    hits = sum(1 for pattern in _CLICKBAIT_PATTERNS if pattern.search(text))
    return min(hits / 3, 1.0)


def _sensationalism(text: str) -> float:
    """Compute a sensationalism heuristic score in [0.0, 1.0].

    Four components (weights sum to 1.0):
    - ALL-CAPS ratio among words longer than 3 chars  × 0.30
    - Exclamation mark density (≥ 3 marks = max)      × 0.25
    - Loaded-term lexicon hits (≥ 3 hits = max)        × 0.25
    - Structural clickbait pattern matches             × 0.20
    """
    words = text.split()
    if not words:
        return 0.0

    # Only consider words longer than 3 chars so common acronyms (WHO, FBI,
    # CIA, BBC, CNN, EU, UK) don't inflate the caps signal on legitimate news.
    long_words = [w for w in words if len(w) > 3]
    if long_words:
        caps_ratio = sum(1 for w in long_words if w.isupper()) / len(long_words)
    else:
        caps_ratio = 0.0

    exclamation_score = min(text.count("!") / 3, 1.0)
    term_hits = sum(
        1 for w in words if w.lower().strip(".,!?;:'\"") in _SENSATIONAL_TERMS
    )
    term_score = min(term_hits / 3, 1.0)

    return round(
        caps_ratio * 0.30
        + exclamation_score * 0.25
        + term_score * 0.25
        + _clickbait_score(text) * 0.20,
        4,
    )


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
