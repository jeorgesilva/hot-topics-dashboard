"""Named entity recognition using spaCy (Week 1, Step 2)."""
from __future__ import annotations

from typing import TypedDict

import spacy

from src.nlp.preprocessor import CleanedItem

_nlp: spacy.language.Language | None = None

# de_core_news_lg label set: PER, ORG, LOC, MISC
ENTITY_LABELS = {"PER", "ORG", "LOC", "MISC"}


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
        _nlp = spacy.load("de_core_news_lg")
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

        if ent.label_ == "PER":
            tags["persons"].append(value)
        elif ent.label_ == "ORG":
            tags["organizations"].append(value)
        elif ent.label_ == "LOC":
            tags["locations"].append(value)
        elif ent.label_ == "MISC":
            tags["events"].append(value)

    return tags


def annotate(item: CleanedItem) -> AnnotatedItem:
    """Add NER entity tags to a cleaned item."""
    entities = extract_entities(item["cleaned_text"])
    return {**item, "entities": entities}  # type: ignore[return-value]


def annotate_batch(items: list[CleanedItem]) -> list[AnnotatedItem]:
    """Annotate a list of cleaned items with NER tags."""
    return [annotate(item) for item in items]
