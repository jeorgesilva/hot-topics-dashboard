"""Topic-level composite risk scorer.

Aggregates per-article risk (from article_scorer.py) and group-level NLP
signals into a single composite_risk score (0–1) per topic, then maps it
to a reliability grade (A–F).

Formula (weights sum to 1.0):
    risk = 0.55 * avg_article_risk        ← bundles source trust, sentiment,
                                             sensationalism, attribution vagueness
         + 0.10 * framing_inconsistency   ← how much articles disagree with each other
         + 0.35 * fact_inconsistency      ← NER entity conflicts across articles

Usage:
    python src/scoring/compute_scores.py
    python src/scoring/compute_scores.py --db-path data/dashboard.db
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
import sqlite3

from src.scoring.article_scorer import score_article
from src.scoring.source_trust import score_coverage
from src.utils.db import init_db

logger = logging.getLogger(__name__)

# Composite formula weights — must sum to 1.0
_WEIGHTS: dict[str, float] = {
    "avg_article_risk":      0.55,
    "framing_inconsistency": 0.10,
    "fact_inconsistency":    0.35,
}

assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

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
        fact_inconsistency: NER entity overlap inconsistency (0–1). Defaults to 0.0.

    Returns:
        Composite risk in [0.0, 1.0].
    """
    return (
        _WEIGHTS["avg_article_risk"]      * avg_article_risk
        + _WEIGHTS["framing_inconsistency"] * framing_inconsistency
        + _WEIGHTS["fact_inconsistency"]    * fact_inconsistency
    )


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

    Shows each signal's weighted contribution to composite_risk so the
    dashboard can explain why a topic was flagged.

    Args:
        topic_id: ID of an already-scored topic.
        conn: Active database connection with row_factory=sqlite3.Row.

    Returns:
        Dict with keys: topic_id, composite_risk, grade, and a
        'contributions' sub-dict showing each signal's weighted value.
        Returns an empty dict if the topic has not been scored yet.
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
        "topic_id":       topic_id,
        "composite_risk": round(r["composite_risk"], 4),
        "grade":          grade_topic(r["composite_risk"]),
        "contributions":  contributions,
    }


def compute_composite(conn: sqlite3.Connection) -> int:
    """Fill composite_risk, social_risk, narrative_divergence, and computed_at.

    A topic is skipped if avg_article_risk or framing_inconsistency is NULL,
    meaning run_nlp has not yet run for that topic.

    social_risk is computed from existing social-track signals (social_avg_trust,
    social_avg_sentiment_extremity, social_sensationalism_avg,
    social_attribution_vagueness) by deriving a social_avg_article_risk via
    the article_scorer formula, then applying the same composite formula.
    narrative_divergence = |composite_risk - social_risk|.

    Args:
        conn: Active database connection.

    Returns:
        Number of topics that received a composite_risk score.
    """
    rows = conn.execute(
        """
        SELECT ts.topic_id,
               ts.avg_article_risk, ts.framing_inconsistency,
               ts.coverage_ratio, ts.fact_inconsistency,
               ts.social_avg_trust, ts.social_avg_sentiment_extremity,
               ts.social_sensationalism_avg, ts.social_coverage_ratio,
               ts.social_framing_inconsistency, ts.social_attribution_vagueness,
               ts.social_fact_inconsistency
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
        risk = compute_risk(
            avg_article_risk=row["avg_article_risk"],
            framing_inconsistency=row["framing_inconsistency"],
            fact_inconsistency=row["fact_inconsistency"] or 0.0,
        )

        social_risk: float | None = None
        if (
            row["social_avg_sentiment_extremity"] is not None
            and row["social_sensationalism_avg"] is not None
            and row["social_framing_inconsistency"] is not None
        ):
            social_avg_article_risk = score_article(
                trust_score=row["social_avg_trust"] or 50.0,
                sentiment_extremity=row["social_avg_sentiment_extremity"],
                sensationalism_score=row["social_sensationalism_avg"],
                attribution_vagueness=row["social_attribution_vagueness"] or 0.0,
            )
            social_risk = compute_risk(
                avg_article_risk=social_avg_article_risk,
                framing_inconsistency=row["social_framing_inconsistency"],
                fact_inconsistency=row["social_fact_inconsistency"] or 0.0,
            )

        divergence = (
            round(abs(risk - social_risk), 6) if social_risk is not None else None
        )

        conn.execute(
            """
            UPDATE topic_scores
            SET composite_risk       = ?,
                social_risk          = ?,
                narrative_divergence = ?,
                computed_at          = ?
            WHERE topic_id = ?
            """,
            (
                round(risk, 6),
                round(social_risk, 6) if social_risk is not None else None,
                divergence,
                now,
                row["topic_id"],
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
