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


def score_attribution_vagueness(text: str) -> float:
    """Score how vaguely attributed the claims in a text are.

    Counts matches against eight structural patterns that signal anonymous
    or unverifiable sourcing. Three or more matches saturates the score at
    1.0; zero matches returns 0.0.

    Args:
        text: Raw or cleaned article text (title + description is sufficient).

    Returns:
        Attribution vagueness score in [0.0, 1.0].
    """
    if not text or not text.strip():
        return 0.0
    hits = sum(1 for pattern in _VAGUE_ATTRIBUTION_PATTERNS if pattern.search(text))
    return round(min(hits / _SATURATION_HITS, 1.0), 4)
