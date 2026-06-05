"""Attribution vagueness scorer (Week 3, Person A).

Detects unattributed or vaguely attributed claims — a key misinformation
signal where sensational assertions are laundered through anonymous sources
("experts say", "sources claim") or adverbial hedges ("reportedly",
"allegedly") instead of named, verifiable sources.

A high attribution_vagueness score does NOT mean a story is false; it means
the sourcing is too opaque to verify, which correlates with misinformation
risk when combined with high sensationalism and low trust scores.
"""
from __future__ import annotations

import re

# ── Named-source anchor detection ────────────────────────────────────────────
# When a named PERSON or ORG entity co-occurs with an attribution verb in the
# same sentence, the sourcing is at least partially specific. Each such anchor
# reduces the raw vagueness score, preventing over-penalising articles that mix
# named and anonymous sources (e.g. "laut Quellen, darunter WHO-Chef Tedros…").

_ATTRIBUTION_VERB_LEMMAS: frozenset[str] = frozenset({
    # German
    "sagen", "erklären", "bestätigen", "betonen", "mitteilen", "berichten",
    "äußern", "sprechen", "informieren", "teilen", "ankündigen", "angeben",
    "schreiben", "kommentieren", "hinweisen", "warnen",
    # English
    "say", "state", "announce", "confirm", "explain", "tell", "report",
    "declare", "emphasize", "note", "add", "write", "comment", "warn",
})

_ANCHOR_DISCOUNT_PER_SENT: float = 0.20   # discount per anchored sentence
_ANCHOR_DISCOUNT_CAP: float = 0.40        # maximum total discount


def _named_source_discount(doc) -> float:
    """Discount for sentences where a named PER/ORG appears with an attribution verb.

    Iterates over sentence boundaries in the spaCy doc. If a sentence contains
    both an attribution verb (lemma match) and at least one PER or ORG entity,
    the sourcing in that sentence is considered anchored and a discount is applied.

    Args:
        doc: spaCy Doc with sentence segmentation and NER annotations.

    Returns:
        Discount in [0.0, 0.40].
    """
    discount = 0.0
    sents = list(doc.sents) if doc.has_annotation("SENT_START") else [doc[:]]
    for sent in sents:
        has_verb = any(
            tok.lemma_.lower() in _ATTRIBUTION_VERB_LEMMAS
            for tok in sent
            if tok.pos_ == "VERB"
        )
        if not has_verb:
            continue
        if any(ent.label_ in ("PER", "ORG") for ent in sent.ents):
            discount += _ANCHOR_DISCOUNT_PER_SENT
    return round(min(discount, _ANCHOR_DISCOUNT_CAP), 4)


# ── Vague attribution patterns ────────────────────────────────────────────────
# Compiled once at import time. Each pattern targets a distinct rhetorical
# move: anonymous-plural attribution, adverbial hedges, agentless passives,
# "according to" with no named source, and speculative framing.

_VAGUE_ATTRIBUTION_PATTERNS: tuple[re.Pattern, ...] = (
    # Anonymous plural agents: EN "experts say" / DE "Experten sagen"
    re.compile(
        r"\b(experts?|analysts?|scientists?|researchers?|officials?|authorities|"
        r"insiders?|sources?|investigators?|observers?)\s+"
        r"(say|says|claim|claims|warn|warns|suggest|suggests|"
        r"believe|believes|report|reports|allege|alleges|"
        r"contend|contends|argue|argues|note|notes|indicate|indicates)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(experten?|analysten?|wissenschaftler|forscher|behörden?|"
        r"beamte|insider|quellen?|ermittler|beobachter|fachleute|"
        r"kritiker|vertreter|sprecher)\s+"
        r"(sagen?|sagten?|behaupten?|warnen?|meinen?|berichten?|"
        r"glauben?|erklären?|sprechen|vermuten?|schätzen?|betonen?)\b",
        re.IGNORECASE,
    ),
    # Adverbial epistemic hedges: EN + DE
    re.compile(
        r"\b(reportedly|allegedly|purportedly|supposedly|apparently|"
        r"seemingly|ostensibly|rumored|rumoured)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(angeblich|offenbar|mutmaßlich|vermeintlich|anscheinend|"
        r"scheinbar|gerüchteweise|dem vernehmen nach|berichten zufolge|"
        r"wie verlautet|wie es heißt)\b",
        re.IGNORECASE,
    ),
    # Agentless passive: EN "it is said" / DE "es soll", "es wird berichtet"
    re.compile(
        r"\bit\s+(is|was|has\s+been)\s+"
        r"(said|claimed|reported|believed|alleged|rumored|rumoured|"
        r"thought|understood|suggested|indicated)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bes\s+(soll|wird|wurde|ist|war)\s+"
        r"(gesagt|behauptet|gemeldet|geglaubt|vermutet|berichtet|"
        r"angenommen|nahegelegt|verlautet)\b",
        re.IGNORECASE,
    ),
    # "according to" with vague source: EN + DE "laut Quellen", "Berichten zufolge"
    re.compile(
        r"\baccording\s+to\s+(sources?|officials?|experts?|insiders?|"
        r"reports?|some|many|authorities|unnamed|anonymous)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(laut|zufolge|nach\s+angaben)\s+(quellen?|insidern?|"
        r"experten?|beobachtern?|berichten?|ungenannten?|anonymen?|"
        r"kreisen|informierten\s+kreisen)\b",
        re.IGNORECASE,
    ),
    # Vague collective: EN "some say" / DE "manche sagen", "viele glauben"
    re.compile(
        r"\b(some|many|most|several|people|critics?|opponents?|skeptics?)\s+"
        r"(say|says|believe|believes|think|thinks|claim|claims|"
        r"argue|argues|fear|allege|alleges)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(manche|viele|einige|mehrere|kritiker|gegner|zweifler|"
        r"beobachter|leute|menschen)\s+"
        r"(sagen?|glauben?|denken?|meinen?|behaupten?|befürchten?|"
        r"vermuten?|zweifeln?)\b",
        re.IGNORECASE,
    ),
    # Rumour framing: EN + DE "Gerüchten zufolge"
    re.compile(
        r"\b(rumors?|rumours?|speculation|whispers?|word)\s+"
        r"(has\s+it|suggest|suggests|indicate|indicates|"
        r"point|points|circulate|circulates|spread|spreads)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(gerüchten?|spekulationen?|munkeln?|gemunkelt)\b",
        re.IGNORECASE,
    ),
    # Unattributed evidence: EN "studies show" / DE "Studien zeigen"
    re.compile(
        r"\b(studies|research|data|reports?|documents?)\s+"
        r"(show|shows|suggest|suggests|indicate|indicates|reveal|reveals|"
        r"confirm|confirms|prove|proves)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(studien?|forschungen?|daten|berichten?|dokumente?|untersuchungen?)\s+"
        r"(zeigen?|belegen?|deuten|ergeben?|bestätigen?|beweisen?|nahelegen?)\b",
        re.IGNORECASE,
    ),
    # Withholding identity: EN + DE "eine mit der Sache vertraute Quelle"
    re.compile(
        r"\b(a\s+source|someone|a\s+person)\s+(familiar\s+with|close\s+to|"
        r"with\s+knowledge\s+of|who\s+asked\s+not\s+to\s+be\s+named|"
        r"speaking\s+on\s+condition\s+of\s+anonymity)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(eine?\s+quelle|jemand|eine?\s+person)\s+"
        r"(die\s+nicht\s+genannt\s+werden\s+(will|möchte)|"
        r"aus\s+(informierten\s+)?kreisen|unter\s+zusicherung\s+der\s+anonymität|"
        r"mit\s+kenntnissen?\s+(der|über)\s+die\s+sache)\b",
        re.IGNORECASE,
    ),
)

# Three pattern hits saturates the score at 1.0, matching the convention used
# in sentiment._clickbait_score() for consistency across scoring modules.
_SATURATION_HITS: int = 3


def score_attribution_vagueness(text: str, nlp=None) -> float:
    """Score how vaguely attributed the claims in a text are.

    Counts matches against structural patterns that signal anonymous or
    unverifiable sourcing. Three or more matches saturates the raw score at
    1.0; zero matches returns 0.0.

    When a loaded spaCy model is passed via `nlp`, a named-source discount is
    applied: for each sentence where a named PERSON or ORG co-occurs with an
    attribution verb the vagueness score is reduced (up to 0.40). This prevents
    over-penalising articles that mix anonymous and verifiable sources.

    Args:
        text: Raw or cleaned article text (title + description is sufficient).
        nlp: Optional loaded spaCy Language model. When provided, enables the
            NER-based named-source discount. Pass None for regex-only scoring.

    Returns:
        Attribution vagueness score in [0.0, 1.0].
    """
    if not text or not text.strip():
        return 0.0
    hits = sum(1 for pattern in _VAGUE_ATTRIBUTION_PATTERNS if pattern.search(text))
    raw_score = min(hits / _SATURATION_HITS, 1.0)
    if nlp is None or raw_score == 0.0:
        return round(raw_score, 4)
    doc = nlp(text[:1500])  # cap length to keep parse time predictable
    discount = _named_source_discount(doc)
    return round(max(0.0, raw_score - discount), 4)
