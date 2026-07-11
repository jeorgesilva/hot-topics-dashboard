"""Check-worthy claim extraction (Fase 5, v1 heuristic).

Given an article's cleaned_text, picks out sentences that plausibly make a
verifiable factual assertion — a number, a date, or a direct attribution of
an event to someone/something — while discarding sentences that read as
opinion, speculation, or hedging.

This is a v1 heuristic, not a trained classifier: it matches curated German
marker word/phrase lists against sentence text, plus regexes for numbers and
dates. It will both over- and under-select in ways a trained model would
avoid (e.g. it cannot tell "12 Verletzte" the number from "Kapitel 12" the
citation). Precision/recall tuning against labelled data is future work —
see notebooks/, per this project's testing convention.
"""
from __future__ import annotations

import re
from typing import TypedDict

from src.nlp.ner import extract_entities

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MIN_SENTENCE_LEN = 20
_MAX_CLAIMS_PER_ARTICLE = 5

# Sentences containing any of these are treated as opinion/speculation/hedging
# and are never check-worthy, regardless of what else they contain.
_OPINION_MARKERS: tuple[str, ...] = (
    "glaubt", "glauben", "glaubte",
    "meint", "meinen", "meinte",
    "vermutet", "vermuten", "vermutete",
    "könnte", "könnten", "könnten",
    "dürfte", "dürften",
    "sollte", "sollten",
    "möglicherweise", "vielleicht", "wahrscheinlich", "angeblich",
    "spekuliert", "spekulation",
    "gerüchten zufolge", "es wird gemunkelt", "man geht davon aus",
    "kritisiert", "kritisierte",
    "warnt vor", "warnte vor",
    "befürchtet", "befürchtete",
    "hofft", "hoffte",
    "fordert", "forderte",
)

# Sentences containing any of these (and no opinion marker) are treated as
# direct factual attribution of an event, even without a number/date.
_FACTUAL_MARKERS: tuple[str, ...] = (
    "bestätigte", "bestätigt",
    "erklärte", "erklärt",
    "gab bekannt", "gaben bekannt",
    "kündigte an", "kündigten an",
    "meldete", "meldet", "meldeten",
    "berichtete", "berichtet",
    "teilte mit", "teilten mit",
    "sagte", "sagten",
    "zufolge",
    "laut ",
    "gab an", "gaben an",
    "verkündete", "verkündeten",
    "wurde mitgeteilt", "wurden mitgeteilt",
)

_DATE_RE = re.compile(
    r"\b\d{1,2}\.\s?(?:Januar|Februar|März|April|Mai|Juni|Juli|August"
    r"|September|Oktober|November|Dezember)(?:\s?\d{4})?\b"
    r"|\b\d{1,2}\.\d{1,2}\.\d{2,4}\b"
    r"|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d")


class Claim(TypedDict):
    text: str
    has_number: bool
    has_date: bool
    entities: list[str]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, dropping fragments shorter than the floor."""
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return [s for s in sentences if len(s) >= _MIN_SENTENCE_LEN]


def _is_check_worthy(sentence: str) -> bool:
    """Return True if a sentence plausibly makes a verifiable factual claim.

    A sentence is check-worthy if it contains a number/date or a direct
    factual-attribution marker, and does NOT contain any opinion/speculation
    marker (opinion markers veto regardless of other content).
    """
    lower = sentence.lower()
    if any(marker in lower for marker in _OPINION_MARKERS):
        return False
    if _NUMBER_RE.search(sentence):
        return True
    if _DATE_RE.search(sentence):
        return True
    return any(marker in lower for marker in _FACTUAL_MARKERS)


def _entity_strings(sentence: str) -> list[str]:
    """Flatten spaCy PER/ORG/LOC entities for a sentence into a plain list."""
    tags = extract_entities(sentence)
    entities: list[str] = []
    for bucket in ("persons", "organizations", "locations"):
        entities.extend(tags[bucket])
    return entities


def extract_claims(cleaned_text: str, max_claims: int = _MAX_CLAIMS_PER_ARTICLE) -> list[Claim]:
    """Extract up to `max_claims` check-worthy claims from an article's text.

    Args:
        cleaned_text: The article's cleaned_text (title + description + body).
        max_claims: Cap on the number of claims returned per article.

    Returns:
        List of Claim dicts, in document order, capped at max_claims.
    """
    claims: list[Claim] = []
    for sentence in _split_sentences(cleaned_text):
        if len(claims) >= max_claims:
            break
        if not _is_check_worthy(sentence):
            continue
        claims.append(
            Claim(
                text=sentence,
                has_number=bool(_NUMBER_RE.search(sentence)),
                has_date=bool(_DATE_RE.search(sentence)),
                entities=_entity_strings(sentence),
            )
        )
    return claims
