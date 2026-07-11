"""Topic-level two-tier risk scorer (Fase 7).

Computes two independent per-topic risk numbers instead of blending
everything into one:

  linguistic_only_risk   — the "old" composite formula (sensationalism,
                            attribution vagueness, framing divergence,
                            NLI fact_inconsistency). Always computable once
                            run_nlp has scored a topic; evidence-independent.
  evidence_grounded_risk — fraction of RAG-verified claims (Fase 5,
                            claim_verifications) that were "refuted" among
                            claims with a definite verdict (supported ∪
                            refuted — excludes not_enough_evidence). NULL
                            when the topic has zero claims with a verdict.
  evidence_coverage      — fraction of the topic's checked claims that had a
                            definite verdict (i.e. were NOT not_enough_evidence).
  overall_confidence     — 'high' / 'medium' / 'low', derived from
                            evidence_coverage and the Fase 6 source-reliability
                            confidence of the topic's article domains.

`topic_scores.composite_risk` remains the single sortable/flaggable number
(dashboard ranking, misinformation banner, and the History queries in
CLAUDE.md), but its derivation is now an explicit, documented combination
rule (compute_overall_risk) instead of an implicit blend — see
src/scoring/weights.py::EVIDENCE_COVERAGE_THRESHOLD.

linguistic_only_risk formula (weights from
src/scoring/weights.py::COMPOSITE_RISK_WEIGHTS, sum to 1.0):
    risk = avg_article_risk * w["avg_article_risk"]            ← bundles source trust,
                                                                   sentiment, sensationalism,
                                                                   attribution vagueness
         + framing_inconsistency * w["framing_inconsistency"]  ← how much articles disagree
         + fact_inconsistency * w["fact_inconsistency"]         ← NLI-detected contradictions
                                                                   (src/scoring/contradiction.py)

Usage:
    python src/scoring/compute_scores.py
    python src/scoring/compute_scores.py --db-path data/dashboard.db
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
import sqlite3

from src.scoring.source_reliability import resolve_reliability
from src.scoring.source_trust import _domain_from_url, score_coverage
from src.scoring.weights import (
    CONFIDENCE_BAND_HIGH,
    CONFIDENCE_BAND_MEDIUM,
    EVIDENCE_COVERAGE_THRESHOLD,
)
from src.scoring.weights import COMPOSITE_RISK_WEIGHTS as _WEIGHTS
from src.utils.db import init_db

logger = logging.getLogger(__name__)

_MISINFO_THRESHOLD = 0.50  # composite_risk above this = likely misinformation


def compute_risk(
    avg_article_risk: float,
    framing_inconsistency: float,
    fact_inconsistency: float = 0.0,
) -> float:
    """Apply the composite risk formula.

    All inputs must be in [0, 1].
    Returns a risk score in [0.0, 1.0] where 1.0 = highest risk.

    Args:
        avg_article_risk: Mean article_risk_score across all articles in the topic.
            Bundles source trust, sentiment extremity, sensationalism, and
            attribution vagueness at the per-article level.
        framing_inconsistency: Cosine-distance-based framing divergence (0–1).
            Measures how much articles in the topic disagree with each other.
        fact_inconsistency: Proportion of cross-tier sentence pairs an NLI
            model classifies as contradiction (0–1). Defaults to 0.0.

    Returns:
        Composite risk in [0.0, 1.0].
    """
    return (
        _WEIGHTS["avg_article_risk"]      * avg_article_risk
        + _WEIGHTS["framing_inconsistency"] * framing_inconsistency
        + _WEIGHTS["fact_inconsistency"]    * fact_inconsistency
    )


def compute_evidence_signals(
    topic_id: int, conn: sqlite3.Connection
) -> tuple[float | None, float]:
    """Derive evidence_grounded_risk and evidence_coverage from claim_verifications.

    Args:
        topic_id: ID of the topic in the topics table.
        conn: Active database connection.

    Returns:
        (evidence_grounded_risk, evidence_coverage). evidence_grounded_risk is
        the fraction of claims with a definite verdict (supported ∪ refuted)
        that were refuted; None when the topic has no claims with a definite
        verdict (including when it has zero claims at all).
        evidence_coverage is the fraction of all checked claims that had a
        definite verdict (0.0 when the topic has zero claims).
    """
    rows = conn.execute(
        "SELECT verdict FROM claim_verifications WHERE topic_id = ?", (topic_id,)
    ).fetchall()
    total = len(rows)
    if total == 0:
        return None, 0.0

    verdicts = [row["verdict"] for row in rows]
    decided = [v for v in verdicts if v != "not_enough_evidence"]
    coverage = len(decided) / total

    if not decided:
        return None, round(coverage, 4)

    refuted = sum(1 for v in decided if v == "refuted")
    return round(refuted / len(decided), 4), round(coverage, 4)


def compute_overall_risk(
    linguistic_only_risk: float,
    evidence_grounded_risk: float | None,
    evidence_coverage: float,
) -> float:
    """Combine the two risk tiers into the single sortable/flaggable number.

    Explicit combination rule (src/scoring/weights.py::EVIDENCE_COVERAGE_THRESHOLD):
    use evidence_grounded_risk once enough claims have a definite verdict to
    trust it over the linguistic-only signal; otherwise fall back to
    linguistic_only_risk.

    Args:
        linguistic_only_risk: Always-available linguistic risk in [0, 1].
        evidence_grounded_risk: Evidence-grounded risk in [0, 1], or None.
        evidence_coverage: Fraction of checked claims with a definite verdict.

    Returns:
        The combined risk in [0.0, 1.0].
    """
    if evidence_grounded_risk is not None and evidence_coverage > EVIDENCE_COVERAGE_THRESHOLD:
        return evidence_grounded_risk
    return linguistic_only_risk


def average_source_confidence(topic_id: int, conn: sqlite3.Connection) -> float | None:
    """Mean Fase 6 resolve_reliability() confidence across a topic's article domains.

    Args:
        topic_id: ID of the topic in the topics table.
        conn: Active database connection.

    Returns:
        Mean confidence in [0, 1], or None if the topic has no articles.
    """
    rows = conn.execute(
        """
        SELECT ri.url, ri.source
        FROM topic_sources ts
        JOIN raw_items ri ON ri.id = ts.item_id
        WHERE ts.topic_id = ?
        """,
        (topic_id,),
    ).fetchall()
    if not rows:
        return None

    confidences = []
    for row in rows:
        domain = _domain_from_url(row["url"]) or row["source"]
        confidences.append(resolve_reliability(domain, conn)["confidence"])

    return sum(confidences) / len(confidences)


def compute_overall_confidence(
    evidence_coverage: float, avg_source_confidence: float | None
) -> str:
    """Map combined evidence + source-reliability confidence to a display band.

    Averages evidence_coverage (how many claims got a definite verdict) with
    the topic's mean Fase 6 source-reliability confidence, then buckets the
    result via src/scoring/weights.py::CONFIDENCE_BAND_HIGH/MEDIUM.

    Args:
        evidence_coverage: Fraction of checked claims with a definite verdict.
        avg_source_confidence: Mean resolve_reliability() confidence across
            the topic's article domains, or None if unavailable.

    Returns:
        'high', 'medium', or 'low'.
    """
    combined = (
        evidence_coverage
        if avg_source_confidence is None
        else (evidence_coverage + avg_source_confidence) / 2.0
    )
    if combined >= CONFIDENCE_BAND_HIGH:
        return "high"
    if combined >= CONFIDENCE_BAND_MEDIUM:
        return "medium"
    return "low"


def grade_topic(risk: float) -> str:
    """Map a composite risk score to a reliability grade.

    Args:
        risk: Composite risk in [0.0, 1.0].

    Returns:
        Letter grade: A (most reliable) → F (least reliable).
    """
    reliability = 1.0 - risk
    if reliability >= 0.80:
        return "A"
    if reliability >= 0.60:
        return "B"
    if reliability >= 0.40:
        return "C"
    if reliability >= 0.20:
        return "D"
    return "F"


def explain_score(topic_id: int, conn: sqlite3.Connection) -> dict:
    """Return a per-signal contribution breakdown for a scored topic.

    Shows each linguistic signal's weighted contribution to
    linguistic_only_risk so the dashboard can explain why a topic was
    flagged, alongside the evidence-grounded tier and the combined
    composite_risk / grade.

    Args:
        topic_id: ID of an already-scored topic.
        conn: Active database connection with row_factory=sqlite3.Row.

    Returns:
        Dict with keys: topic_id, composite_risk, grade, linguistic_only_risk,
        evidence_grounded_risk, evidence_coverage, overall_confidence, and a
        'contributions' sub-dict showing each linguistic signal's weighted
        value. Returns an empty dict if the topic has not been scored yet.
    """
    row = conn.execute(
        "SELECT * FROM topic_scores WHERE topic_id = ?", (topic_id,)
    ).fetchone()

    if row is None or row["composite_risk"] is None:
        return {}

    r = dict(row)
    contributions = {
        "article_risk":          round(_WEIGHTS["avg_article_risk"] * (r["avg_article_risk"] or 0.0), 4),
        "framing_inconsistency": round(_WEIGHTS["framing_inconsistency"] * (r["framing_inconsistency"] or 0.0), 4),
        "fact_inconsistency":    round(_WEIGHTS["fact_inconsistency"] * (r["fact_inconsistency"] or 0.0), 4),
    }
    return {
        "topic_id":               topic_id,
        "composite_risk":         round(r["composite_risk"], 4),
        "grade":                  grade_topic(r["composite_risk"]),
        "linguistic_only_risk":   r["linguistic_only_risk"],
        "evidence_grounded_risk": r["evidence_grounded_risk"],
        "evidence_coverage":      r["evidence_coverage"],
        "overall_confidence":     r["overall_confidence"],
        "contributions":          contributions,
    }


def compute_composite(conn: sqlite3.Connection) -> int:
    """Fill the two-tier risk scores and computed_at.

    A topic is skipped if avg_article_risk or framing_inconsistency is NULL,
    meaning run_nlp has not yet run for that topic.

    Two-tier scoring (Fase 7):
      linguistic_only_risk   — the "old" composite formula (compute_risk).
      evidence_grounded_risk — from claim_verifications (compute_evidence_signals).
      evidence_coverage      — fraction of checked claims with a definite verdict.
      composite_risk         — combine_overall_risk(linguistic_only_risk,
                                evidence_grounded_risk, evidence_coverage):
                                the single sortable/flaggable number.
      overall_confidence     — compute_overall_confidence(evidence_coverage,
                                average_source_confidence(topic_id)).

    Args:
        conn: Active database connection.

    Returns:
        Number of topics that received a composite_risk score.
    """
    rows = conn.execute(
        """
        SELECT ts.topic_id,
               ts.avg_article_risk, ts.framing_inconsistency,
               ts.coverage_ratio, ts.fact_inconsistency
        FROM topic_scores ts
        JOIN topics t ON t.id = ts.topic_id
        WHERE ts.avg_article_risk IS NOT NULL
          AND ts.framing_inconsistency IS NOT NULL
          AND COALESCE(t.run_id, -1) = COALESCE((SELECT MAX(run_id) FROM topics), -1)
        """
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    scored = 0

    for row in rows:
        topic_id = row["topic_id"]

        linguistic_only_risk = compute_risk(
            avg_article_risk=row["avg_article_risk"],
            framing_inconsistency=row["framing_inconsistency"],
            fact_inconsistency=row["fact_inconsistency"] or 0.0,
        )
        evidence_grounded_risk, evidence_coverage = compute_evidence_signals(topic_id, conn)
        risk = compute_overall_risk(
            linguistic_only_risk=linguistic_only_risk,
            evidence_grounded_risk=evidence_grounded_risk,
            evidence_coverage=evidence_coverage,
        )
        overall_confidence = compute_overall_confidence(
            evidence_coverage=evidence_coverage,
            avg_source_confidence=average_source_confidence(topic_id, conn),
        )

        conn.execute(
            """
            UPDATE topic_scores
            SET composite_risk         = ?,
                linguistic_only_risk   = ?,
                evidence_grounded_risk = ?,
                evidence_coverage      = ?,
                overall_confidence     = ?,
                computed_at            = ?
            WHERE topic_id = ?
            """,
            (
                round(risk, 6),
                round(linguistic_only_risk, 6),
                evidence_grounded_risk,
                evidence_coverage,
                overall_confidence,
                now,
                topic_id,
            ),
        )
        scored += 1

    conn.commit()
    logger.info("Composite risk computed for %d topics.", scored)
    return scored


def score_all_topics(conn: sqlite3.Connection) -> dict[str, int]:
    """Orchestrate the full scoring pipeline for all topics.

    Step 1: Compute and persist coverage metrics (coverage_ratio, avg_trust).
    Step 2: Compute composite_risk for topics where avg_article_risk is present.

    Args:
        conn: Active database connection.

    Returns:
        Summary dict: {'coverage_scored': n, 'composite_scored': n}.
    """
    coverage_n = score_coverage(conn)
    composite_n = compute_composite(conn)
    logger.info(
        "score_all_topics: %d topics coverage-scored, %d composite-scored.",
        coverage_n,
        composite_n,
    )
    return {"coverage_scored": coverage_n, "composite_scored": composite_n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute topic risk scores.")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database. Default: data/dashboard.db.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    conn = init_db(args.db_path)
    summary = score_all_topics(conn)
    conn.close()

    print(f"Coverage metrics scored : {summary['coverage_scored']} topics")
    print(f"Composite risk computed : {summary['composite_scored']} topics")


if __name__ == "__main__":
    main()
