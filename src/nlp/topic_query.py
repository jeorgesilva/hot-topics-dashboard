"""Build NewsAPI search queries from a topic cluster (Week 1, Step 3)."""
from __future__ import annotations

import re

from src.nlp.keywords import extract_keywords
from src.nlp.ner import AnnotatedItem

_PRIORITY_BUCKETS = ("persons", "organizations", "locations")

# Tokens that strongly indicate a news/media organization. ORG entities whose
# lowercased text contains any of these words are excluded from the query so
# that publication names pulled from RSS description snippets don't pollute
# the NewsAPI search.
_MEDIA_TOKENS = frozenset({
    "news", "times", "post", "tribune", "daily", "journal", "herald",
    "gazette", "press", "media", "broadcasting", "television", "radio",
    "channel", "network", "magazine", "weekly", "wire",
})

# Well-known broadcast/wire abbreviations that carry no media-domain word
# but are unambiguously news outlets in this context.
_KNOWN_OUTLETS = frozenset({
    "cnn", "bbc", "fox", "nbc", "abc", "cbs", "pbs", "sky",
    "msnbc", "cnbc", "npr", "espn", "cspan", "hln", "ap",
})


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

    # Source names from the cluster (lowercased) are excluded from ORG entities
    # because RSS description snippets often cite the publishing outlet by name.
    _source_lower: set[str] = {
        (item.get("source") or "").lower() for item in items
    } - {""}

    def _normalise_tokens(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"[a-z0-9]+", text.lower()))

    def _token_sequence_in(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
        if not needle or len(needle) > len(haystack):
            return False
        return any(
            haystack[i:i + len(needle)] == needle
            for i in range(len(haystack) - len(needle) + 1)
        )

    _source_tokens: tuple[tuple[str, ...], ...] = tuple(
        _normalise_tokens(src) for src in _source_lower
    )

    def _is_media_org(entity: str) -> bool:
        e = entity.lower()
        entity_tokens = _normalise_tokens(e)
        if any(
            entity_tokens == src_tokens
            or _token_sequence_in(entity_tokens, src_tokens)
            or _token_sequence_in(src_tokens, entity_tokens)
            for src_tokens in _source_tokens
        ):
            return True
        tokens = set(e.split())
        if tokens & _MEDIA_TOKENS:
            return True
        # Strip trailing digits from each token (catches "ABC7", "FOX 5")
        alpha_tokens = {re.sub(r"\d+$", "", t) for t in tokens} - {""}
        return bool(alpha_tokens & _KNOWN_OUTLETS)

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

    for item in items:
        for bucket in _PRIORITY_BUCKETS:
            for entity in item["entities"][bucket]:
                if word_count >= max_terms:
                    break
                # Restrict media-outlet filtering to ORG entities so locations
                # and persons containing ambiguous tokens are preserved.
                if bucket == "organizations" and _is_media_org(entity):
                    continue
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
