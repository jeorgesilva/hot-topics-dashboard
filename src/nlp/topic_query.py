"""Build NewsAPI search queries from a topic cluster (Week 1, Step 3)."""
from __future__ import annotations

from src.nlp.keywords import extract_keywords
from src.nlp.ner import AnnotatedItem


def build_topic_query(items: list[AnnotatedItem], max_terms: int = 5) -> str:
    """Produce a NewsAPI search query from a cluster of annotated RSS items.

    NER entities (persons, orgs, locations) are prioritised over TF-IDF
    keywords. Terms are deduplicated case-insensitively and capped at
    max_terms so the query stays focused.

    Args:
        items: Annotated items belonging to the same topic cluster.
        max_terms: Maximum number of terms in the output query.

    Returns:
        A space-joined query string, e.g. ``"EU AI regulation framework"``.
        Returns an empty string when no items are provided.
    """
    if not items:
        return ""

    seen: set[str] = set()
    terms: list[str] = []

    def _add(term: str) -> None:
        normalised = term.lower().strip()
        if len(normalised) > 1 and normalised not in seen:
            seen.add(normalised)
            terms.append(term.strip())

    for item in items:
        for entity_list in item["entities"].values():
            for entity in entity_list:
                _add(entity)

    if len(terms) < max_terms:
        keyword_lists = extract_keywords(items, top_n=max_terms)
        for kw_list in keyword_lists:
            for kw in kw_list:
                _add(kw)
                if len(terms) >= max_terms:
                    break
            if len(terms) >= max_terms:
                break

    return " ".join(" ".join(terms).split()[:max_terms])
