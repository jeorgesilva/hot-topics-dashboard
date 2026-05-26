"""Topic-level composite risk scorer.

Aggregates Person-B coverage signals and Person-A NLP signals into a
single composite_risk score (0–1) per topic, then maps it to a
reliability grade (A–F).

Formula (weights sum to 1.0):
    risk = 0.25 * (1 - avg_trust / 100)
         + 0.20 * avg_sentiment_extremity
         + 0.20 * (1 - coverage_ratio)
         + 0.15 * framing_inconsistency
         + 0.10 * sensationalism_avg
         + 0.05 * attribution_vagueness
         + 0.05 * fact_inconsistency

Usage:
    python src/scoring/compute_scores.py
    python src/scoring/compute_scores.py --db-path data/dashboard.db
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
import sqlite3

from src.scoring.source_trust import score_coverage
from src.utils.db import init_db

logger = logging.getLogger(__name__)

# Composite formula weights — must sum to 1.0
_WEIGHTS: dict[str, float] = {
    "avg_trust":              0.25,
    "avg_sentiment_extremity": 0.20,
    "coverage_ratio":         0.20,
    "framing_inconsistency":  0.15,
    "sensationalism_avg":     0.10,
    "attribution_vagueness":  0.05,
    "fact_inconsistency":     0.05,
}

assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

_MISINFO_THRESHOLD = 0.50  # composite_risk above this = likely misinformation


def compute_risk(
    avg_trust: float,
    avg_sentiment_extremity: float,
    coverage_ratio: float,
    framing_inconsistency: float,
    sensationalism_avg: float,
    attribution_vagueness: float = 0.0,
    fact_inconsistency: float = 0.0,
) -> float:
    """Apply the composite risk formula.

    All input signals must be in [0, 1] except avg_trust which is [0, 100].
    Returns a risk score in [0.0, 1.0] where 1.0 = highest risk.

    Args:
        avg_trust: Mean source trust score (0–100) from source_trust.py.
        avg_sentiment_extremity: Mean sentiment extremity (0–1) from Person A.
        coverage_ratio: Fraction of unique credible domains (0–1).
        framing_inconsistency: Cosine-distance-based framing divergence (0–1) from Person A.
        sensationalism_avg: Mean sensationalism score (0–1) from Person A.
        attribution_vagueness: Vague attribution density (0–1) from Person A. Defaults to 0.0.
        fact_inconsistency: NER entity overlap inconsistency (0–1) from Person A. Defaults to 0.0.

    Returns:
        Composite risk in [0.0, 1.0].
    """
    return (
        _WEIGHTS["avg_trust"]               * (1.0 - avg_trust / 100.0)
        + _WEIGHTS["avg_sentiment_extremity"] * avg_sentiment_extremity
        + _WEIGHTS["coverage_ratio"]          * (1.0 - coverage_ratio)
        + _WEIGHTS["framing_inconsistency"]   * framing_inconsistency
        + _WEIGHTS["sensationalism_avg"]      * sensationalism_avg
        + _WEIGHTS["attribution_vagueness"]   * attribution_vagueness
        + _WEIGHTS["fact_inconsistency"]      * fact_inconsistency
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

    Shows each signal's weighted contribution to the composite_risk so
    the dashboard can explain why a topic was flagged.

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
    avg_trust = r["avg_trust"] or 0.0
    contributions = {
        "source_distrust":       round(_WEIGHTS["avg_trust"] * (1.0 - avg_trust / 100.0), 4),
        "sentiment_extremity":   round(_WEIGHTS["avg_sentiment_extremity"] * (r["avg_sentiment_extremity"] or 0.0), 4),
        "low_credible_coverage": round(_WEIGHTS["coverage_ratio"] * (1.0 - (r["coverage_ratio"] or 0.0)), 4),
        "framing_inconsistency": round(_WEIGHTS["framing_inconsistency"] * (r["framing_inconsistency"] or 0.0), 4),
        "sensationalism":        round(_WEIGHTS["sensationalism_avg"] * (r["sensationalism_avg"] or 0.0), 4),
        "attribution_vagueness": round(_WEIGHTS["attribution_vagueness"] * (r["attribution_vagueness"] or 0.0), 4),
        "fact_inconsistency":    round(_WEIGHTS["fact_inconsistency"] * (r["fact_inconsistency"] or 0.0), 4),
    }
    return {
        "topic_id":      topic_id,
        "composite_risk": round(r["composite_risk"], 4),
        "grade":          grade_topic(r["composite_risk"]),
        "contributions":  contributions,
    }


def compute_composite(conn: sqlite3.Connection) -> int:
    """Fill composite_risk and computed_at for topics with all signals present.

    A topic is skipped if any of the three Person-A columns
    (avg_sentiment_extremity, sensationalism_avg, framing_inconsistency)
    are NULL — those are filled by Person A's pipeline and may not yet
    be available.

    Args:
        conn: Active database connection.

    Returns:
        Number of topics that received a composite_risk score.
    """
    rows = conn.execute(
        """
        SELECT topic_id, avg_trust, avg_sentiment_extremity,
               coverage_ratio, framing_inconsistency, sensationalism_avg,
               attribution_vagueness, fact_inconsistency
        FROM topic_scores
        WHERE avg_trust IS NOT NULL
          AND avg_sentiment_extremity IS NOT NULL
          AND sensationalism_avg IS NOT NULL
          AND framing_inconsistency IS NOT NULL
        """
    ).fetchall()

    now = datetime.now(timezone.utc).isoformat()
    scored = 0

    for row in rows:
        risk = compute_risk(
            avg_trust=row["avg_trust"],
            avg_sentiment_extremity=row["avg_sentiment_extremity"],
            coverage_ratio=row["coverage_ratio"] or 0.0,
            framing_inconsistency=row["framing_inconsistency"],
            sensationalism_avg=row["sensationalism_avg"],
            attribution_vagueness=row["attribution_vagueness"] or 0.0,
            fact_inconsistency=row["fact_inconsistency"] or 0.0,
        )
        conn.execute(
            """
            UPDATE topic_scores
            SET composite_risk = ?, computed_at = ?
            WHERE topic_id = ?
            """,
            (round(risk, 6), now, row["topic_id"]),
        )
        scored += 1

    conn.commit()
    logger.info("Composite risk computed for %d topics.", scored)
    return scored


def score_all_topics(conn: sqlite3.Connection) -> dict[str, int]:
    """Orchestrate the full scoring pipeline for all topics.

    Step 1: Compute and persist coverage metrics (Person B signals).
    Step 2: Compute composite_risk for topics where Person A signals are present.

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
