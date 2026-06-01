"""Text cleaning and tokenization pipeline using spaCy (Week 1, Step 1)."""
from __future__ import annotations

import html
import re
import unicodedata
from typing import TypedDict

import spacy

_nlp: spacy.language.Language | None = None


class RawItem(TypedDict):
    id: str
    title: str
    description: str | None
    source: str
    url: str
    platform: str
    timestamp: str
    engagement: dict


class CleanedItem(RawItem):
    cleaned_text: str
    tokens: list[str]
    lemmas: list[str]


def _get_nlp() -> spacy.language.Language:
    """Load and cache the spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("de_core_news_lg", disable=["parser", "ner"])
    return _nlp


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form and strip non-printable characters."""
    normalized = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cc")


def clean_text(text: str) -> str:
    """Decode HTML entities, strip tags, normalize unicode, collapse whitespace."""
    text = html.unescape(text)
    text = strip_html(text)
    text = normalize_unicode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_lemmatize(text: str) -> tuple[list[str], list[str]]:
    """Return (tokens, lemmas) for alphabetic, non-stop-word tokens."""
    nlp = _get_nlp()
    doc = nlp(text.lower())
    tokens = [t.text for t in doc if t.is_alpha and not t.is_stop]
    lemmas = [t.lemma_ for t in doc if t.is_alpha and not t.is_stop]
    return tokens, lemmas


def preprocess(item: RawItem) -> CleanedItem:
    """Clean and tokenize a single raw item.

    Combines title, description, and body_text (when available) into a single
    cleaned_text field, then runs tokenization and lemmatization via spaCy.
    Body text is the full scraped article body from article_fetcher; it provides
    substantially more signal for NLP scoring than title+description alone.
    """
    raw_text = item["title"]
    if item.get("description"):
        raw_text = f"{raw_text} {item['description']}"
    body = item.get("body_text")  # type: ignore[typeddict-item]
    if body:
        raw_text = f"{raw_text} {body}"

    cleaned = clean_text(raw_text)
    tokens, lemmas = tokenize_and_lemmatize(cleaned)

    return CleanedItem(**item, cleaned_text=cleaned, tokens=tokens, lemmas=lemmas)


def preprocess_batch(items: list[RawItem]) -> list[CleanedItem]:
    """Preprocess a list of raw items."""
    return [preprocess(item) for item in items]
