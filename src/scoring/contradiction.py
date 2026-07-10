"""Cross-tier factual contradiction detection via NLI (Fase 4).

Replaces the previous Jaccard entity-overlap fact_inconsistency metric
(src/scoring/framing.py::_entity_overlap_score, removed), which penalised
articles for simply naming different people/organizations — a false
positive on ordinary journalistic diversity, not on factual disagreement.

fact_inconsistency is now the proportion of high-trust/low-trust sentence
pairs that a multilingual NLI model classifies as "contradiction" above a
confidence threshold. Running NLI on every sentence-pair combination is
quadratic and too slow for a 10-20 article topic, so candidate pairs are
pre-filtered by sentence-embedding similarity (reusing the same model as
framing.py) and by shared named entities (reusing src/nlp/ner.py), then
capped at _MAX_PAIRS before the expensive NLI forward pass.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.nlp.embeddings import get_model
from src.nlp.ner import extract_entities

if TYPE_CHECKING:
    from src.scoring.sentiment import ScoredArticle

_pipeline = None

_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

# Cost controls: NLI forward passes are the expensive step, so only the
# _MAX_PAIRS most promising sentence pairs (by embedding similarity /
# shared entities) are ever sent to the model per topic.
_MAX_SENTENCES_PER_TIER = 60
_MAX_PAIRS = 15
_SIM_FLOOR = 0.35
_MIN_SENTENCE_LEN = 20
_CONTRADICTION_CONFIDENCE_THRESHOLD = 0.60

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _get_pipeline():
    """Lazy-load the HuggingFace NLI text-classification pipeline."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline(
            "text-classification",
            model=_NLI_MODEL,
            top_k=None,
            tokenizer_kwargs={"truncation": True, "max_length": 256},
        )
    return _pipeline


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a punctuation heuristic.

    Drops short fragments (headlines, bylines) below _MIN_SENTENCE_LEN
    characters, since these rarely carry a checkable factual claim.
    """
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return [s for s in sentences if len(s) >= _MIN_SENTENCE_LEN]


def _collect_sentences(articles: list[ScoredArticle], limit: int) -> list[str]:
    """Gather sentences across a tier's articles, capped at `limit` total."""
    sentences: list[str] = []
    for article in articles:
        text = article.get("cleaned_text") or article.get("title") or ""
        sentences.extend(_split_sentences(text))
        if len(sentences) >= limit:
            break
    return sentences[:limit]


def _sentence_entities(text: str) -> set[str]:
    """Return lowercased PERSON/ORG/LOC entity strings mentioned in a sentence."""
    tags = extract_entities(text)
    entities: set[str] = set()
    for bucket in ("persons", "organizations", "locations"):
        for e in tags[bucket]:
            entities.add(e.lower().strip())
    return entities


def _cosine(a, b) -> float:
    import numpy as np
    dot = float(np.dot(a, b))
    mag_a = float(np.linalg.norm(a))
    mag_b = float(np.linalg.norm(b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _select_candidate_pairs(
    high_sents: list[str],
    low_sents: list[str],
    model,
) -> list[tuple[str, str]]:
    """Pick the _MAX_PAIRS most promising sentence pairs for NLI scoring.

    Two-stage filter to keep the quadratic step cheap and the expensive
    NLI step bounded:
    1. Embedding similarity >= _SIM_FLOOR — discards pairs that are about
       clearly unrelated content (NLI on unrelated sentences is meaningless
       and would just add noise/cost).
    2. Among the surviving pairs, those that share a named entity are
       ranked first (a shared entity makes a pair a much stronger
       candidate for a genuine factual claim/counter-claim), then by
       descending similarity. The top _MAX_PAIRS are returned.

    Args:
        high_sents: Candidate sentences from the high-trust tier.
        low_sents: Candidate sentences from the low-trust tier.
        model: Loaded SentenceTransformer instance.

    Returns:
        Up to _MAX_PAIRS (high_sentence, low_sentence) tuples.
    """
    if not high_sents or not low_sents:
        return []

    high_emb = model.encode(high_sents, batch_size=32, show_progress_bar=False)
    low_emb = model.encode(low_sents, batch_size=32, show_progress_bar=False)

    above_floor: list[tuple[float, int, int]] = []
    for i in range(len(high_sents)):
        for j in range(len(low_sents)):
            sim = _cosine(high_emb[i], low_emb[j])
            if sim >= _SIM_FLOOR:
                above_floor.append((sim, i, j))

    if not above_floor:
        return []

    entity_cache: dict[tuple[str, int], set[str]] = {}

    def _entities_for(tier: str, sents: list[str], idx: int) -> set[str]:
        key = (tier, idx)
        if key not in entity_cache:
            entity_cache[key] = _sentence_entities(sents[idx])
        return entity_cache[key]

    ranked: list[tuple[bool, float, int, int]] = []
    for sim, i, j in above_floor:
        shares_entity = bool(
            _entities_for("h", high_sents, i) & _entities_for("l", low_sents, j)
        )
        ranked.append((shares_entity, sim, i, j))

    ranked.sort(key=lambda t: (not t[0], -t[1]))
    top = ranked[:_MAX_PAIRS]
    return [(high_sents[i], low_sents[j]) for _, _, i, j in top]


def _is_contradiction(raw: list[dict]) -> bool:
    """Return True if the NLI pipeline's "contradiction" score clears the threshold.

    Args:
        raw: A single pair's result from the top_k=None pipeline, e.g.
            [{"label": "entailment", "score": 0.1}, {"label": "contradiction", "score": 0.8}, ...].
    """
    scores = {entry["label"].lower(): entry["score"] for entry in raw}
    return scores.get("contradiction", 0.0) >= _CONTRADICTION_CONFIDENCE_THRESHOLD


def compute_fact_inconsistency(
    high_articles: list[ScoredArticle],
    low_articles: list[ScoredArticle],
) -> float:
    """Proportion of evaluated cross-tier sentence pairs classified as contradiction.

    Selects up to _MAX_PAIRS candidate sentence pairs between the high-trust
    and low-trust tiers (see _select_candidate_pairs), runs each through an
    NLI model, and returns the fraction labelled "contradiction" above
    _CONTRADICTION_CONFIDENCE_THRESHOLD. Returns 0.0 when no tier has
    extractable sentences or no candidate pair clears the similarity floor —
    i.e. when there isn't enough signal to assess contradiction, the topic
    is not penalised (same fail-open philosophy as the rest of this module).

    Args:
        high_articles: Articles in the high-trust tier.
        low_articles: Articles in the low-trust tier.

    Returns:
        fact_inconsistency in [0.0, 1.0].
    """
    high_sents = _collect_sentences(high_articles, _MAX_SENTENCES_PER_TIER)
    low_sents = _collect_sentences(low_articles, _MAX_SENTENCES_PER_TIER)
    if not high_sents or not low_sents:
        return 0.0

    model = get_model()
    pairs = _select_candidate_pairs(high_sents, low_sents, model)
    if not pairs:
        return 0.0

    pipe = _get_pipeline()
    inputs = [{"text": h, "text_pair": l} for h, l in pairs]
    results = pipe(inputs)

    contradictions = sum(1 for raw in results if _is_contradiction(raw))
    return round(contradictions / len(pairs), 4)
