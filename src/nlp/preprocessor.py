"""Text cleaning and tokenization pipeline using spaCy (Week 1, Step 1)."""
from __future__ import annotations

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
        _nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])
    return _nlp


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFC form and strip non-printable characters."""
    normalized = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cc")


def clean_text(text: str) -> str:
    """Strip HTML, normalize unicode, and collapse whitespace."""
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

    Combines title and description into a single cleaned_text field,
    then runs tokenization and lemmatization via spaCy.
    """
    raw_text = item["title"]
    if item.get("description"):
        raw_text = f"{raw_text} {item['description']}"

    cleaned = clean_text(raw_text)
    tokens, lemmas = tokenize_and_lemmatize(cleaned)

    return CleanedItem(**item, cleaned_text=cleaned, tokens=tokens, lemmas=lemmas)


def preprocess_batch(items: list[RawItem]) -> list[CleanedItem]:
    """Preprocess a list of raw items."""
    return [preprocess(item) for item in items]
