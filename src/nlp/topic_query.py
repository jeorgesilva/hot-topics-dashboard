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
    word_count = 0

    def _add(term: str) -> None:
        nonlocal word_count
        normalised = term.lower().strip()
        new_words = len(term.split())
        if len(normalised) > 1 and normalised not in seen and word_count + new_words <= max_terms:
            seen.add(normalised)
            terms.append(term.strip())
            word_count += new_words

    _PRIORITY_BUCKETS = ("persons", "organizations", "locations")
    for item in items:
        for bucket in _PRIORITY_BUCKETS:
            for entity in item["entities"][bucket]:
                if word_count >= max_terms:
                    break
                _add(entity)

    if word_count < max_terms:
        try:
            keyword_lists = extract_keywords(items, top_n=max_terms)
        except ValueError as exc:
            if "empty vocabulary" in str(exc).lower():
                keyword_lists = []
            else:
                raise
        for kw_list in keyword_lists:
            for kw in kw_list:
                _add(kw)
                if word_count >= max_terms:
                    break
            if word_count >= max_terms:
                break

    return " ".join(terms)
