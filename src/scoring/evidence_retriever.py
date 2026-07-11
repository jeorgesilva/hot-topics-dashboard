"""Evidence retrieval for claim verification (Fase 5).

For a given extracted claim, looks for corroborating/refuting evidence in
three tiers, stopping at the first one that returns a result:

  1. Google Fact Check Tools API (GOOGLE_FACT_CHECK_API_KEY) — purpose-built
     fact-check database; the strongest signal when it has a match.
  2. Wikidata SPARQL — looks up a Wikidata entity description for one of the
     claim's named entities. Reuses the request pattern from
     src/scoring/domain_resolver.py::_wikidata_signal, but that function
     interpolates a raw domain string into its SPARQL query body with no
     escaping — a query-injection weakness. Here entity labels are escaped
     via _sparql_escape() before being embedded as a quoted literal, and the
     query binds an exact label match rather than an unescaped CONTAINS/regex
     substring, so untrusted claim text cannot break out of the literal.
     This tier is inherently weak evidence: it corroborates "who/what is X"
     (entity descriptions), not numeric or event-specific assertions.
  3. Corpus fallback — other articles already collected for the same topic.
     Not external evidence, but useful when nothing else is available: if
     another article in the topic makes a closely related (high embedding
     similarity) statement, that sentence is returned as evidence.
"""
from __future__ import annotations

import logging
import sqlite3
import urllib.parse
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

_FACT_CHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
_FACT_CHECK_TIMEOUT = 8
_FACT_CHECK_QUERY_MAX_LEN = 200  # API rejects overly long free-text queries

_WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
_WIKIDATA_TIMEOUT = 10
_WIKIDATA_MAX_ENTITIES_TRIED = 3

# Generic Wikidata item types whose "description" is not actual evidence
# (e.g. a disambiguation page for an ambiguous entity string like "Union" or
# "dpa") — feeding these to the NLI model produces misleadingly confident
# verdicts against unrelated text, so they are skipped in favour of the next
# candidate entity.
_WIKIDATA_LOW_VALUE_DESCRIPTIONS: tuple[str, ...] = (
    "Wikimedia-Begriffsklärungsseite",
    "Begriffsklärungsseite",
    "Wikimedia-Liste",
    "Wikimedia-Kategorie",
)

_CORPUS_SIM_FLOOR = 0.55
_CORPUS_MIN_SENTENCE_LEN = 20


def _cosine(a, b) -> float:
    import numpy as np
    dot = float(np.dot(a, b))
    mag_a = float(np.linalg.norm(a))
    mag_b = float(np.linalg.norm(b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


@dataclass
class Evidence:
    text: str
    url: str
    source_type: str  # "google_factcheck" | "wikidata" | "corpus"
    rating: str | None = None  # textualRating, only set for google_factcheck


# ---------------------------------------------------------------------------
# Tier 1: Google Fact Check Tools API
# ---------------------------------------------------------------------------

def _query_google_fact_check(claim_text: str) -> Evidence | None:
    """Query the Fact Check Tools API for a matching fact-check review.

    Returns None (never raises) on missing key, HTTP error, or no match —
    same fail-open philosophy as domain_resolver.py's signal fetchers.
    """
    from src.utils.config import GOOGLE_FACT_CHECK_API_KEY

    if not GOOGLE_FACT_CHECK_API_KEY:
        return None

    try:
        resp = requests.get(
            _FACT_CHECK_ENDPOINT,
            params={
                "query": claim_text[:_FACT_CHECK_QUERY_MAX_LEN],
                "languageCode": "de",
                "key": GOOGLE_FACT_CHECK_API_KEY,
            },
            timeout=_FACT_CHECK_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("evidence_retriever: Fact Check API failed: %s", exc)
        return None

    claims = data.get("claims") or []
    if not claims:
        return None

    reviews = claims[0].get("claimReview") or []
    if not reviews:
        return None

    review = reviews[0]
    text = review.get("title") or claims[0].get("text") or ""
    url = review.get("url") or ""
    rating = review.get("textualRating")
    if not text or not url:
        return None

    return Evidence(text=text, url=url, source_type="google_factcheck", rating=rating)


# ---------------------------------------------------------------------------
# Tier 2: Wikidata SPARQL (sanitized)
# ---------------------------------------------------------------------------

def _sparql_escape(value: str) -> str:
    """Escape a string for safe embedding inside a SPARQL string literal.

    Escapes backslashes and double quotes per SPARQL string-literal syntax,
    and strips newlines (which would otherwise let injected text break out
    of the literal and inject new triple patterns/clauses).
    """
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _query_wikidata(entities: list[str]) -> Evidence | None:
    """Look up a Wikidata entity description for one of the claim's entities.

    Tries up to _WIKIDATA_MAX_ENTITIES_TRIED entities, exact-label matching
    each as an escaped SPARQL string literal (never interpolated unescaped),
    and returns the first entity found with a non-empty description.
    """
    for entity in entities[:_WIKIDATA_MAX_ENTITIES_TRIED]:
        entity = entity.strip()
        if not entity:
            continue
        escaped = _sparql_escape(entity)
        sparql = f"""
        SELECT ?item ?itemDescription WHERE {{
          ?item rdfs:label "{escaped}"@de .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "de,en". }}
        }} LIMIT 1
        """
        url = (
            _WIKIDATA_ENDPOINT
            + "?query="
            + urllib.parse.quote(sparql.strip())
            + "&format=json"
        )
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": "hot-topics-dashboard/1.0 (research project)"},
                timeout=_WIKIDATA_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("evidence_retriever: Wikidata lookup failed for %r: %s", entity, exc)
            continue

        bindings = data.get("results", {}).get("bindings") or []
        if not bindings:
            continue

        binding = bindings[0]
        item_url = binding.get("item", {}).get("value", "")
        description = binding.get("itemDescription", {}).get("value", "")
        if not item_url or not description:
            continue
        if description in _WIKIDATA_LOW_VALUE_DESCRIPTIONS:
            continue

        return Evidence(text=description, url=item_url, source_type="wikidata")

    return None


# ---------------------------------------------------------------------------
# Tier 3: corpus fallback (other articles in the same topic)
# ---------------------------------------------------------------------------

def _query_corpus(
    claim_text: str,
    conn: sqlite3.Connection,
    topic_id: int,
    exclude_item_id: str | None,
) -> Evidence | None:
    """Search other articles in the same topic for a closely related sentence.

    Not external evidence — the fallback of last resort when neither the
    Fact Check API nor Wikidata returned anything.
    """
    import re

    rows = conn.execute(
        """
        SELECT ri.id, ri.url, ri.cleaned_text
        FROM raw_items ri
        JOIN topic_sources tsrc ON tsrc.item_id = ri.id
        WHERE tsrc.topic_id = ? AND ri.cleaned_text IS NOT NULL
        """,
        (topic_id,),
    ).fetchall()

    sentence_split_re = re.compile(r"(?<=[.!?])\s+")
    candidates: list[tuple[str, str]] = []  # (sentence, url)
    for row in rows:
        if exclude_item_id is not None and row["id"] == exclude_item_id:
            continue
        text = row["cleaned_text"] or ""
        for sentence in sentence_split_re.split(text):
            sentence = sentence.strip()
            if len(sentence) >= _CORPUS_MIN_SENTENCE_LEN:
                candidates.append((sentence, row["url"]))

    if not candidates:
        return None

    from src.nlp.embeddings import get_model

    model = get_model()
    claim_emb = model.encode([claim_text], show_progress_bar=False)[0]
    sentences = [c[0] for c in candidates]
    sent_embs = model.encode(sentences, batch_size=32, show_progress_bar=False)

    best_sim = -1.0
    best_idx = -1
    for i, emb in enumerate(sent_embs):
        sim = _cosine(claim_emb, emb)
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_idx == -1 or best_sim < _CORPUS_SIM_FLOOR:
        return None

    sentence, url = candidates[best_idx]
    return Evidence(text=sentence, url=url, source_type="corpus")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_evidence(
    claim_text: str,
    entities: list[str] | None = None,
    conn: sqlite3.Connection | None = None,
    topic_id: int | None = None,
    exclude_item_id: str | None = None,
) -> Evidence | None:
    """Retrieve the best available evidence for a claim.

    Tries Google Fact Check → Wikidata → corpus fallback, in order, and
    returns the first hit. Returns None if no tier finds anything.

    Args:
        claim_text: The extracted claim sentence.
        entities: Named entities extracted from the claim (feeds the
            Wikidata tier). Optional — that tier is skipped if empty.
        conn: Active DB connection, required for the corpus fallback tier.
        topic_id: Topic the claim's article belongs to, required for the
            corpus fallback tier.
        exclude_item_id: Article ID to exclude from corpus search results
            (the article the claim itself came from).

    Returns:
        Evidence from the first tier that finds a match, or None.
    """
    evidence = _query_google_fact_check(claim_text)
    if evidence is not None:
        return evidence

    evidence = _query_wikidata(entities or [])
    if evidence is not None:
        return evidence

    if conn is not None and topic_id is not None:
        evidence = _query_corpus(claim_text, conn, topic_id, exclude_item_id)
        if evidence is not None:
            return evidence

    return None
