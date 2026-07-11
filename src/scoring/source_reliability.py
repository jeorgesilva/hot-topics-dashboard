"""Truth-discovery-style source reliability estimation (Fase 6).

Treats `domain_resolver.resolve_trust()` (via `source_trust.get_trust_score`)
as a PRIOR — a heuristic estimate available before any claim has ever been
checked against that domain — and builds a POSTERIOR on top of it from the
observed history of `claim_verifications` (Fase 5): each "supported" verdict
nudges a domain's reliability up, each "refuted" verdict nudges it down,
"not_enough_evidence" leaves the score untouched. `resolve_reliability()`
blends the two, weighted by how much posterior evidence has accumulated.

Simplifications relative to a reference truth-discovery algorithm
(e.g. TruthFinder / Sums-Products, Yin et al. 2008):

  - No cross-source / cross-claim graph propagation. A real truth-discovery
    algorithm jointly estimates claim truthfulness and source trustworthiness
    by iterating over the whole source×claim bipartite graph until
    convergence (source trust depends on the claims it made being true,
    claim truth depends on the trust of sources asserting it). This module
    only tracks each domain's own verdict history — it never lets domain A's
    reliability influence domain B's, and it never revisits the NLI verdict
    for a claim once `claim_verifier.py` has produced it.
  - The "EM" here is a single-pass exponentially time-decayed weighted moving
    average per domain, not an iterative fixpoint. Each new verdict updates
    that domain's running (weighted_sum, weight_total) once; there is no
    re-estimation loop over historical verdicts.
  - Confidence is a hand-tuned heuristic (prior tier + accumulated posterior
    weight saturating at `_POSTERIOR_SATURATION_WEIGHT` verdicts), not a
    formally derived posterior variance.

These tradeoffs keep the update O(1) per verdict and safe to run
incrementally as `claim_verifications` grows, at the cost of not capturing
inter-source corroboration effects a full truth-discovery pass would.
"""
from __future__ import annotations

import csv
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CSV_PATH = _PROJECT_ROOT / "config" / "source_trust.csv"

# Verdict → signal in [-1, 1]. not_enough_evidence deliberately maps to 0.0
# AND is given zero update weight below, so it never moves the score.
_VERDICT_SIGNAL: dict[str, float] = {
    "supported": 1.0,
    "refuted": -1.0,
    "not_enough_evidence": 0.0,
}

# Older verdicts count for less. Half-life in days: a verdict's contribution
# to the running average halves every _DECAY_HALFLIFE_DAYS.
_DECAY_HALFLIFE_DAYS: float = 30.0
_DECAY_RATE: float = math.log(2) / _DECAY_HALFLIFE_DAYS

# Accumulated verdict weight at which posterior confidence saturates to 1.0.
_POSTERIOR_SATURATION_WEIGHT: float = 5.0

# Prior confidence tiers — how much we trust the PRIOR score itself, based on
# how it was obtained. Verified/low come from source_trust.csv's `confidence`
# column; "dynamic" means domain_resolver has scored it at least once;
# "unknown" means neither — a flat fallback score with no real signal at all.
_PRIOR_CONFIDENCE_VERIFIED: float = 0.9
_PRIOR_CONFIDENCE_LOW: float = 0.6
_PRIOR_CONFIDENCE_DYNAMIC: float = 0.4
_PRIOR_CONFIDENCE_UNKNOWN: float = 0.2


def _normalize_domain(domain: str) -> str:
    return domain.strip().lower().removeprefix("www.")


def _load_confidence_db(csv_path: Path = _CSV_PATH) -> dict[str, str]:
    """Load domain → confidence ('verified'/'low') from source_trust.csv."""
    db: dict[str, str] = {}
    if not csv_path.exists():
        return db
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            domain = _normalize_domain(row.get("domain", ""))
            confidence = row.get("confidence", "").strip().lower()
            if domain and confidence:
                db[domain] = confidence
    return db


# Module-level cache — loaded once per process, mirrors source_trust.py.
_CONFIDENCE_DB: dict[str, str] = _load_confidence_db()


def init_cache(conn: sqlite3.Connection) -> None:
    """Create the source_reliability tables if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_reliability (
            domain            TEXT PRIMARY KEY,
            weighted_sum      REAL NOT NULL DEFAULT 0.0,
            weight_total      REAL NOT NULL DEFAULT 0.0,
            verdict_count     INTEGER NOT NULL DEFAULT 0,
            reliability_score REAL NOT NULL DEFAULT 50.0,
            updated_at        TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source_reliability_applied (
            claim_verification_id INTEGER PRIMARY KEY
        )
    """)
    conn.commit()


class ReliabilityState(TypedDict):
    domain: str
    weighted_sum: float
    weight_total: float
    verdict_count: int
    reliability_score: float
    updated_at: str


class ReliabilityEstimate(TypedDict):
    domain: str
    trust_score: float
    confidence: float
    prior_score: float
    prior_confidence: float
    posterior_score: float | None
    posterior_weight: float
    verdict_count: int


def _prior_confidence(domain: str, conn: sqlite3.Connection) -> float:
    """How much to trust the PRIOR score for this domain.

    verified CSV entry > low-confidence CSV entry > domain_resolver has
    already scored it dynamically at least once > no signal whatsoever.
    """
    key = _normalize_domain(domain)
    tier = _CONFIDENCE_DB.get(key)
    if tier == "verified":
        return _PRIOR_CONFIDENCE_VERIFIED
    if tier == "low":
        return _PRIOR_CONFIDENCE_LOW
    row = conn.execute(
        "SELECT 1 FROM domain_trust_cache WHERE domain = ?", (key,)
    ).fetchone()
    if row is not None:
        return _PRIOR_CONFIDENCE_DYNAMIC
    return _PRIOR_CONFIDENCE_UNKNOWN


def get_reliability_state(
    conn: sqlite3.Connection, domain: str
) -> ReliabilityState | None:
    """Return the raw persisted posterior state for a domain, or None."""
    key = _normalize_domain(domain)
    row = conn.execute(
        """
        SELECT domain, weighted_sum, weight_total, verdict_count,
               reliability_score, updated_at
        FROM source_reliability WHERE domain = ?
        """,
        (key,),
    ).fetchone()
    if row is None:
        return None
    return ReliabilityState(
        domain=row["domain"],
        weighted_sum=row["weighted_sum"],
        weight_total=row["weight_total"],
        verdict_count=row["verdict_count"],
        reliability_score=row["reliability_score"],
        updated_at=row["updated_at"],
    )


def _posterior_weight(weight_total: float) -> float:
    return min(weight_total / _POSTERIOR_SATURATION_WEIGHT, 1.0)


def update_reliability(
    conn: sqlite3.Connection,
    domain: str,
    verdict: str,
    confidence: float,
    at: str | None = None,
) -> ReliabilityState:
    """Fold one claim verdict into a domain's reliability estimate.

    Time-decayed weighted moving average: the domain's existing
    (weighted_sum, weight_total) is decayed by elapsed time since its last
    update, then the new verdict is added with weight = confidence (0 for
    not_enough_evidence, so it decays existing history but never moves the
    score on its own).

    Args:
        conn: Active database connection with source_reliability table present.
        domain: Domain the verdict's article was published on.
        verdict: One of "supported", "refuted", "not_enough_evidence"
            (src.scoring.claim_verifier.Verdict).
        confidence: NLI confidence for the verdict, in [0, 1].
        at: ISO timestamp of the verdict (defaults to now). Callers replaying
            historical claim_verifications rows should pass their checked_at.

    Returns:
        The updated ReliabilityState.
    """
    key = _normalize_domain(domain)
    at_dt = datetime.fromisoformat(at) if at else datetime.now(timezone.utc)
    if at_dt.tzinfo is None:
        at_dt = at_dt.replace(tzinfo=timezone.utc)

    prev = get_reliability_state(conn, key)
    if prev is None:
        weighted_sum, weight_total, verdict_count = 0.0, 0.0, 0
    else:
        prev_at = datetime.fromisoformat(prev["updated_at"])
        if prev_at.tzinfo is None:
            prev_at = prev_at.replace(tzinfo=timezone.utc)
        elapsed_days = max((at_dt - prev_at).total_seconds() / 86400.0, 0.0)
        decay = math.exp(-_DECAY_RATE * elapsed_days)
        weighted_sum = prev["weighted_sum"] * decay
        weight_total = prev["weight_total"] * decay
        verdict_count = prev["verdict_count"]

    signal = _VERDICT_SIGNAL.get(verdict, 0.0)
    weight = 0.0 if verdict == "not_enough_evidence" else max(float(confidence), 0.0)
    weighted_sum += signal * weight
    weight_total += weight
    verdict_count += 1

    reliability_raw = weighted_sum / weight_total if weight_total > 0 else 0.0
    reliability_score = min(max(50.0 + 50.0 * reliability_raw, 0.0), 100.0)

    state = ReliabilityState(
        domain=key,
        weighted_sum=weighted_sum,
        weight_total=weight_total,
        verdict_count=verdict_count,
        reliability_score=round(reliability_score, 4),
        updated_at=at_dt.isoformat(),
    )
    conn.execute(
        """
        INSERT INTO source_reliability
            (domain, weighted_sum, weight_total, verdict_count, reliability_score, updated_at)
        VALUES (:domain, :weighted_sum, :weight_total, :verdict_count, :reliability_score, :updated_at)
        ON CONFLICT(domain) DO UPDATE SET
            weighted_sum      = excluded.weighted_sum,
            weight_total      = excluded.weight_total,
            verdict_count     = excluded.verdict_count,
            reliability_score = excluded.reliability_score,
            updated_at        = excluded.updated_at
        """,
        state,
    )
    conn.commit()
    return state


def resolve_reliability(domain: str, conn: sqlite3.Connection) -> ReliabilityEstimate:
    """Blend the heuristic PRIOR with the observed-verdict POSTERIOR.

    Few accumulated verdicts → trust_score stays close to the prior
    (domain_resolver / source_trust). Many accumulated verdicts → trust_score
    shifts toward the observed reliability_score. `confidence` is explicit
    about uncertainty: it is only low when *both* the prior is weak (domain
    absent from source_trust.csv and never dynamically resolved) and the
    posterior has little history — i.e. a domain about which nothing is
    known, as opposed to one with a known-average heuristic trust score.

    Args:
        domain: Domain to resolve, e.g. 'example.de'.
        conn: Active database connection with source_reliability,
            domain_trust_cache tables present (src.utils.db.init_db).

    Returns:
        ReliabilityEstimate with the blended trust_score and its confidence.
    """
    from src.scoring.source_trust import get_trust_score

    key = _normalize_domain(domain)
    prior_score = get_trust_score(key, conn=conn)
    prior_conf = _prior_confidence(key, conn)

    state = get_reliability_state(conn, key)
    if state is None or state["weight_total"] <= 0:
        posterior_weight = 0.0
        posterior_score: float | None = None
        verdict_count = state["verdict_count"] if state else 0
    else:
        posterior_weight = _posterior_weight(state["weight_total"])
        posterior_score = state["reliability_score"]
        verdict_count = state["verdict_count"]

    blended_score = (
        prior_score
        if posterior_score is None
        else prior_score * (1.0 - posterior_weight) + posterior_score * posterior_weight
    )
    confidence = 1.0 - (1.0 - prior_conf) * (1.0 - posterior_weight)

    return ReliabilityEstimate(
        domain=key,
        trust_score=round(blended_score, 2),
        confidence=round(confidence, 4),
        prior_score=round(prior_score, 2),
        prior_confidence=round(prior_conf, 4),
        posterior_score=posterior_score,
        posterior_weight=round(posterior_weight, 4),
        verdict_count=verdict_count,
    )


def sync_from_claim_verifications(conn: sqlite3.Connection) -> int:
    """Fold every not-yet-applied claim_verifications row into source_reliability.

    Idempotent: each processed claim_verifications.id is recorded in
    source_reliability_applied, so re-running only picks up verdicts written
    since the previous sync.

    Args:
        conn: Active database connection with claim_verifications,
            raw_items, source_reliability, and source_reliability_applied
            tables present.

    Returns:
        Number of claim_verifications rows applied in this call.
    """
    from src.scoring.source_trust import _domain_from_url

    rows = conn.execute(
        """
        SELECT cv.id, cv.verdict, cv.confidence, cv.checked_at, ri.url
        FROM claim_verifications cv
        JOIN raw_items ri ON ri.id = cv.item_id
        WHERE cv.id NOT IN (SELECT claim_verification_id FROM source_reliability_applied)
        ORDER BY cv.checked_at
        """
    ).fetchall()

    for row in rows:
        domain = _domain_from_url(row["url"])
        if domain:
            update_reliability(
                conn, domain, row["verdict"], row["confidence"] or 0.0, at=row["checked_at"]
            )
        conn.execute(
            "INSERT OR IGNORE INTO source_reliability_applied (claim_verification_id) VALUES (?)",
            (row["id"],),
        )

    conn.commit()
    if rows:
        logger.info("source_reliability: applied %d new claim verdicts.", len(rows))
    return len(rows)
