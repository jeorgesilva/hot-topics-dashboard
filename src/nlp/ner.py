"""Named entity recognition using spaCy (Week 1, Step 2)."""
from __future__ import annotations

from typing import TypedDict

import spacy

from src.nlp.preprocessor import CleanedItem

_nlp: spacy.language.Language | None = None

ENTITY_LABELS = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "NORP"}


class EntityTags(TypedDict):
    persons: list[str]
    organizations: list[str]
    locations: list[str]
    events: list[str]


class AnnotatedItem(CleanedItem):
    entities: EntityTags


def _get_nlp() -> spacy.language.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> EntityTags:
    """Run spaCy NER and return structured entity tags."""
    nlp = _get_nlp()
    doc = nlp(text)

    tags: EntityTags = {"persons": [], "organizations": [], "locations": [], "events": []}
    seen: set[str] = set()

    for ent in doc.ents:
        if ent.label_ not in ENTITY_LABELS:
            continue
        value = ent.text.strip()
        key = f"{ent.label_}:{value.lower()}"
        if key in seen:
            continue
        seen.add(key)

        if ent.label_ == "PERSON":
            tags["persons"].append(value)
        elif ent.label_ in {"ORG", "NORP"}:
            tags["organizations"].append(value)
        elif ent.label_ in {"GPE", "LOC"}:
            tags["locations"].append(value)
        elif ent.label_ == "EVENT":
            tags["events"].append(value)

    return tags


def annotate(item: CleanedItem) -> AnnotatedItem:
    """Add NER entity tags to a cleaned item."""
    entities = extract_entities(item["cleaned_text"])
    return AnnotatedItem(**item, entities=entities)


def annotate_batch(items: list[CleanedItem]) -> list[AnnotatedItem]:
    """Annotate a list of cleaned items with NER tags."""
    return [annotate(item) for item in items]
