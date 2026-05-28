"""NLP pipeline runner: score all topics with Person-A signals.

Reads articles for each topic from the DB, runs sentiment + sensationalism
scoring, framing inconsistency, and attribution vagueness, then writes
the aggregated signals into topic_scores. Call compute_scores.py afterward
to compute the composite risk.

Usage:
    python -m src.scoring.run_nlp
    python -m src.scoring.run_nlp --db-path data/dashboard.db
"""
from __future__ import annotations

import argparse
import logging
import statistics
from datetime import datetime, timezone

from src.scoring.attribution import score_attribution_vagueness
from src.scoring.framing import compute_framing
from src.scoring.sentiment import score_articles
from src.scoring.source_trust import get_trust_score, _domain_from_url
from src.utils.db import get_connection, init_db

logger = logging.getLogger(__name__)


def _load_topic_articles(conn, topic_id: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT ri.id, ri.title, ri.description, ri.source, ri.url,
               ri.platform, ri.timestamp, ri.engagement_json,
               ri.cleaned_text
        FROM topic_sources ts
        JOIN raw_items ri ON ri.id = ts.item_id
        WHERE ts.topic_id = ?
        """,
        (topic_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def run_nlp_pipeline(db_path=None) -> dict:
    """Score all topics with NLP signals and persist to topic_scores."""
    conn = init_db(db_path)
    topic_ids = [r["id"] for r in conn.execute("SELECT id FROM topics").fetchall()]

    if not topic_ids:
        logger.warning("No topics found in DB.")
        conn.close()
        return {"topics_scored": 0}

    logger.info("Running NLP pipeline on %d topics...", len(topic_ids))
    scored = 0

    for topic_id in topic_ids:
        articles = _load_topic_articles(conn, topic_id)
        if not articles:
            logger.info("  topic %d: no articles, skipping", topic_id)
            continue

        logger.info("  topic %d: scoring %d articles...", topic_id, len(articles))

        # strip DB-cached cleaned_text so preprocess() can recompute fresh
        raw_articles = [{k: v for k, v in a.items() if k != "cleaned_text"} for a in articles]

        # Sentiment + sensationalism via RoBERTa
        scored_articles = score_articles(raw_articles)

        avg_sentiment_extremity = statistics.mean(
            a["sentiment_extremity"] for a in scored_articles
        )
        sensationalism_avg = statistics.mean(
            a["sensationalism_score"] for a in scored_articles
        )

        # Framing inconsistency via sentence-transformers
        trust_map = {
            a["source"]: get_trust_score(_domain_from_url(a["url"]) or a["source"])
            for a in scored_articles
        }
        framing_result = compute_framing(scored_articles, trust_map)
        framing_inconsistency = framing_result["framing_inconsistency"]
        fact_inconsistency = framing_result["fact_inconsistency"]

        # Attribution vagueness — average over article texts
        attr_scores = [
            score_attribution_vagueness(a.get("cleaned_text") or a["title"])
            for a in scored_articles
        ]
        attribution_vagueness = statistics.mean(attr_scores)

        # Update cleaned_text on raw_items
        for a in scored_articles:
            if a.get("cleaned_text"):
                conn.execute(
                    "UPDATE raw_items SET cleaned_text = ? WHERE id = ?",
                    (a["cleaned_text"], a["id"]),
                )

        # Upsert NLP signals into topic_scores
        conn.execute(
            """
            INSERT INTO topic_scores (topic_id, avg_sentiment_extremity,
                sensationalism_avg, framing_inconsistency, fact_inconsistency,
                attribution_vagueness)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(topic_id) DO UPDATE SET
                avg_sentiment_extremity = excluded.avg_sentiment_extremity,
                sensationalism_avg      = excluded.sensationalism_avg,
                framing_inconsistency   = excluded.framing_inconsistency,
                fact_inconsistency      = excluded.fact_inconsistency,
                attribution_vagueness   = excluded.attribution_vagueness
            """,
            (
                topic_id,
                round(avg_sentiment_extremity, 4),
                round(sensationalism_avg, 4),
                round(framing_inconsistency, 4),
                round(fact_inconsistency, 4),
                round(attribution_vagueness, 4),
            ),
        )
        conn.commit()
        scored += 1
        logger.info(
            "    ✓ sentiment_ext=%.3f  sensationalism=%.3f  framing=%.3f  attribution=%.3f",
            avg_sentiment_extremity, sensationalism_avg, framing_inconsistency, attribution_vagueness,
        )

    conn.close()
    logger.info("NLP pipeline complete: %d topics scored.", scored)
    return {"topics_scored": scored}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NLP scoring for all topics.")
    parser.add_argument("--db-path", default=None, help="SQLite DB path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    result = run_nlp_pipeline(args.db_path)
    print(f"Topics NLP-scored: {result['topics_scored']}")


if __name__ == "__main__":
    main()
